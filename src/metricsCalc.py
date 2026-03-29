"""
Evaluation script for annotation metrics.

Compares predictions (annotations_folder/)
against ground truth (GroundTruthISI/).

Matching strategy:
  - Two annotations are paired if their quantity spans OVERLAP
    (i.e. the [begin, end) intervals share at least one character).
  - When multiple GT annotations are candidates for the same prediction,
    the one with the greatest overlap is chosen (greedy, best-first).
  - This handles cases where the model includes extra words in the span
    (e.g. GT "1.3 million" vs pred "nearly 1.3 million").

Metrics computed per field:
  - Precision, Recall, F1  — exact character match of .text value.
  - Levenshtein similarity  — 1 - dist / max(len(a), len(b)).
    FN and FP annotations contribute 0.0 to the Levenshtein average
    so the metric is not artificially inflated by missed annotations.

Aggregation:
  - Per-document.
  - Global MICRO: counts pooled across all documents.
  - Global MACRO: average of per-document metrics (each document
    weights equally regardless of annotation count).

Usage:
    python src/metricsCalc.py \
        --gt   ../Data/GroundTruthISI \
        --pred ../Data/annotations_folder \
        --out  ../Data/results.csv

The annotations_folder is a placeholder for the actual directory where the model-generated annotations are stored.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Levenshtein (pure Python, no external dependencies)
# ---------------------------------------------------------------------------

def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1,
                            prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    """1 - dist / max(len(a), len(b)).  Returns 1.0 if both strings are empty."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(a, b) / max_len


# ---------------------------------------------------------------------------
# Field / span helpers
# ---------------------------------------------------------------------------

ALL_FIELDS = ["quantity", "modifier", "unit", "eventDescription", "eventType"]


def get_field_text(ann: dict, field: str):
    """
    Return the .text string for a field, or None if absent.
    eventType is a plain string value (no nested dict).
    """
    val = ann.get(field)
    if val is None:
        return None
    if field == "eventType":
        return str(val) if val else None
    if isinstance(val, dict):
        return val.get("text")
    return None


def get_quantity_span(ann: dict):
    """
    Return (begin, end) of the quantity field.
    Accepts both 'begin'/'start' key names (the model uses 'start').
    Returns (None, None) if the field is absent.
    """
    q = ann.get("quantity")
    if not isinstance(q, dict):
        return (None, None)
    begin = q.get("begin") if q.get("begin") is not None else q.get("start")
    end   = q.get("end")
    return (begin, end)


def span_overlap(b1, e1, b2, e2) -> int:
    """Number of overlapping characters between [b1,e1) and [b2,e2)."""
    if None in (b1, e1, b2, e2):
        return 0
    return max(0, min(e1, e2) - max(b1, b2))


# ---------------------------------------------------------------------------
# File-pair discovery
# ---------------------------------------------------------------------------

def strip_extensions(filename: str) -> str:
    """Remove all extensions: 'a1_1599.txt.json' -> 'a1_1599'."""
    p = Path(filename)
    while p.suffix:
        p = p.with_suffix("")
    return p.name


def discover_pairs(gt_dir: str, pred_dir: str):
    def index_dir(d):
        return {strip_extensions(f): os.path.join(d, f)
                for f in os.listdir(d)
                if f.lower().endswith((".json", ".jason"))}

    gt_files   = index_dir(gt_dir)
    pred_files = index_dir(pred_dir)
    common     = sorted(set(gt_files) & set(pred_files))
    if not common:
        print("WARNING: No matching file pairs found.", file=sys.stderr)
    return [(s, gt_files[s], pred_files[s]) for s in common]


def load_annotations(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data]
    return []


# ---------------------------------------------------------------------------
# Matching — text-exact first, offset tiebreak, overlap fallback
# ---------------------------------------------------------------------------

def get_quantity_text(ann: dict):
    q = ann.get("quantity")
    if not isinstance(q, dict):
        return None
    return q.get("text")


def match_annotations(gt_list: list, pred_list: list):
    """
    Two-phase matching strategy:

    PHASE 1 — exact quantity.text match
      For each pred, find GT annotations sharing the exact same quantity.text.
      - If there is only one such GT candidate → pair immediately.
      - If there are multiple (same word appears twice in the text) → use offset
        overlap to pick the best one. If the pred has no offsets (null), any
        available GT candidate with that text is accepted (first-come order).
      Matching is greedy best-first: candidates are scored as
        (2, overlap_or_1, gi, pi)  so text matches always beat offset matches.

    PHASE 2 — offset overlap fallback (for remaining unmatched annotations)
      For preds not matched in phase 1, try to pair via quantity-span overlap.
      This handles cases where the model captured extra words in the span
      (e.g. pred "nearly 1.3 million" overlaps GT "1.3 million").
      Scored as (1, overlap, gi, pi).

    Both phases use greedy best-first assignment; each annotation is used
    at most once across both phases.

    Returns:
      matched_pairs  : list of (gt_ann, pred_ann)
      unmatched_gt   : list of GT annotations with no prediction
      unmatched_pred : list of predicted annotations with no GT
    """
    # priority: 2 = exact text match, 1 = offset overlap only
    # score tuple: (priority, overlap_score, gi, pi)
    candidates = []

    for pi, pred_ann in enumerate(pred_list):
        pred_text = get_quantity_text(pred_ann)
        pb, pe    = get_quantity_span(pred_ann)
        pred_has_offsets = (pb is not None and pe is not None)

        for gi, gt_ann in enumerate(gt_list):
            gt_text = get_quantity_text(gt_ann)
            gb, ge  = get_quantity_span(gt_ann)

            # --- Phase 1: exact text match ---
            if pred_text is not None and pred_text == gt_text:
                if pred_has_offsets:
                    ov = span_overlap(pb, pe, gb, ge)
                    # Even zero overlap is fine here (text already matched);
                    # use overlap only as a tiebreaker score.
                    tiebreak = ov if ov > 0 else 1
                else:
                    tiebreak = 1   # no offsets → accept any text match
                candidates.append((2, tiebreak, gi, pi))

            # --- Phase 2: offset overlap fallback ---
            # Only added if texts differ (or one is missing),
            # so it cannot steal a pair that should have been text-matched.
            elif pred_has_offsets:
                ov = span_overlap(pb, pe, gb, ge)
                if ov > 0:
                    candidates.append((1, ov, gi, pi))

    # Sort: highest priority first, then highest overlap score
    candidates.sort(key=lambda x: (-x[0], -x[1]))

    used_gt   = set()
    used_pred = set()
    matched_pairs = []

    for _pri, _score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matched_pairs.append((gt_list[gi], pred_list[pi]))

    unmatched_gt   = [ann for i, ann in enumerate(gt_list)   if i not in used_gt]
    unmatched_pred = [ann for i, ann in enumerate(pred_list) if i not in used_pred]

    return matched_pairs, unmatched_gt, unmatched_pred


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class FieldAccumulator:
    """
    Tracks TP/FP/FN and Levenshtein scores for one field in one document.

    Levenshtein policy:
      - Matched pair, texts equal   -> 1.0
      - Matched pair, texts differ  -> lev_similarity (< 1)
      - FN (missed GT annotation)   -> 0.0  (model contributed nothing)
      - FP (spurious prediction)    -> 0.0  (no GT counterpart)
      - Field absent in pred only   -> 0.0
      - Field absent in GT only     -> 0.0
    """
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.lev_scores = []

    def add_match(self, gt_text: str, pred_text: str):
        """Both fields present in a matched annotation pair."""
        if gt_text == pred_text:
            self.tp += 1
        else:
            self.fp += 1
            self.fn += 1
        self.lev_scores.append(levenshtein_similarity(gt_text, pred_text))

    def add_fn_annotation(self):
        """Full annotation missed by the model."""
        self.fn += 1
        self.lev_scores.append(0.0)

    def add_fp_annotation(self):
        """Spurious annotation invented by the model."""
        self.fp += 1
        self.lev_scores.append(0.0)

    def add_fn_field(self):
        """GT field present but model omitted it inside a matched annotation."""
        self.fn += 1
        self.lev_scores.append(0.0)

    def add_fp_field(self):
        """Model field present but GT doesn't have it inside a matched annotation."""
        self.fp += 1
        self.lev_scores.append(0.0)

    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def avg_levenshtein(self) -> float:
        return sum(self.lev_scores) / len(self.lev_scores) if self.lev_scores else 0.0


# ---------------------------------------------------------------------------
# Per-document evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(gt_list: list, pred_list: list) -> dict:
    """Returns {field: FieldAccumulator} for one document."""
    accs = {f: FieldAccumulator() for f in ALL_FIELDS}

    matched_pairs, unmatched_gt, unmatched_pred = match_annotations(gt_list, pred_list)

    for gt_ann, pred_ann in matched_pairs:
        for field in ALL_FIELDS:
            gt_text   = get_field_text(gt_ann,   field)
            pred_text = get_field_text(pred_ann,  field)
            acc = accs[field]
            if gt_text is None and pred_text is None:
                pass
            elif gt_text is None:
                acc.add_fp_field()
            elif pred_text is None:
                acc.add_fn_field()
            else:
                acc.add_match(gt_text, pred_text)

    for gt_ann in unmatched_gt:
        for field in ALL_FIELDS:
            if get_field_text(gt_ann, field) is not None:
                accs[field].add_fn_annotation()

    for pred_ann in unmatched_pred:
        for field in ALL_FIELDS:
            if get_field_text(pred_ann, field) is not None:
                accs[field].add_fp_annotation()

    return accs


# ---------------------------------------------------------------------------
# Global micro accumulator
# ---------------------------------------------------------------------------

class GlobalAccumulator:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.lev_scores = []

    def merge(self, acc: FieldAccumulator):
        self.tp += acc.tp
        self.fp += acc.fp
        self.fn += acc.fn
        self.lev_scores.extend(acc.lev_scores)

    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def avg_levenshtein(self) -> float:
        return sum(self.lev_scores) / len(self.lev_scores) if self.lev_scores else 0.0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def fmt(v):
    return f"{v:.4f}" if v is not None else "  N/A  "


def print_table(title, rows, col_headers):
    print(f"\n{'='*76}")
    print(f"  {title}")
    print(f"{'='*76}")
    col_w, field_w = 12, 20
    header = f"{'Field':<{field_w}}" + "".join(f"{h:>{col_w}}" for h in col_headers)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row[0]:<{field_w}}" + "".join(f"{fmt(v):>{col_w}}" for v in row[1:]))


def save_csv(out_path, doc_results, global_accs, macro_doc):
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "document", "field",
                    "precision", "recall", "f1", "levenshtein",
                    "TP", "FP", "FN"])
        for stem, accs in doc_results:
            for field, acc in accs.items():
                w.writerow(["document", stem, field,
                            acc.precision(), acc.recall(), acc.f1(),
                            acc.avg_levenshtein(),
                            acc.tp, acc.fp, acc.fn])
        for field, acc in global_accs.items():
            w.writerow(["global_micro", "ALL", field,
                        acc.precision(), acc.recall(), acc.f1(),
                        acc.avg_levenshtein(),
                        acc.tp, acc.fp, acc.fn])
        for field, (p, r, f1, lev) in macro_doc.items():
            w.writerow(["global_macro", "ALL", field, p, r, f1, lev, "", "", ""])
    print(f"\nResults saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",   required=True, help="Ground truth directory")
    parser.add_argument("--pred", required=True, help="Predictions directory")
    parser.add_argument("--out",  default=None,  help="Optional CSV output path")
    args = parser.parse_args()

    pairs = discover_pairs(args.gt, args.pred)
    if not pairs:
        print("No matching file pairs found.")
        sys.exit(1)
    print(f"Found {len(pairs)} matching document pair(s).")

    doc_results = []
    global_accs = {f: GlobalAccumulator() for f in ALL_FIELDS}

    for stem, gt_path, pred_path in pairs:
        gt_list   = load_annotations(gt_path)
        pred_list = load_annotations(pred_path)
        accs      = evaluate_pair(gt_list, pred_list)
        doc_results.append((stem, accs))
        for field, acc in accs.items():
            global_accs[field].merge(acc)

    # Macro: average per-document metrics.
    # Documents with no annotations for a field contribute 0.0.
    macro_doc = {}
    for field in ALL_FIELDS:
        ps = [accs[field].precision()       for _, accs in doc_results]
        rs = [accs[field].recall()          for _, accs in doc_results]
        fs = [accs[field].f1()              for _, accs in doc_results]
        ls = [accs[field].avg_levenshtein() for _, accs in doc_results]
        macro_doc[field] = (
            sum(ps) / len(ps),
            sum(rs) / len(rs),
            sum(fs) / len(fs),
            sum(ls) / len(ls),
        )

    # Per-document
    for stem, accs in doc_results:
        rows = [[f, accs[f].precision(), accs[f].recall(), accs[f].f1(),
                 accs[f].avg_levenshtein(), accs[f].tp, accs[f].fp, accs[f].fn]
                for f in ALL_FIELDS]
        print_table(f"Document: {stem}", rows,
                    ["Precision", "Recall", "F1", "Levenshtein", "TP", "FP", "FN"])

    # Global micro
    rows = [[f, global_accs[f].precision(), global_accs[f].recall(),
             global_accs[f].f1(), global_accs[f].avg_levenshtein(),
             global_accs[f].tp, global_accs[f].fp, global_accs[f].fn]
            for f in ALL_FIELDS]
    print_table("GLOBAL MICRO (all annotations pooled)", rows,
                ["Precision", "Recall", "F1", "Levenshtein", "TP", "FP", "FN"])

    # Global macro
    rows = [[f] + list(macro_doc[f]) for f in ALL_FIELDS]
    print_table("GLOBAL MACRO (average over documents)", rows,
                ["Precision", "Recall", "F1", "Levenshtein"])

    if args.out:
        save_csv(args.out, doc_results, global_accs, macro_doc)


if __name__ == "__main__":
    main()