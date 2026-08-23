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

# ---------------------------------------------------------------------------
# Value normalization — compare quantities numerically, not as strings
# ("5 million" == "5,000,000" == "5000000"; "four" == "4")
# ---------------------------------------------------------------------------

import re

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_MULTIPLIERS = {"hundred": 1e2, "thousand": 1e3, "million": 1e6, "billion": 1e9}


def normalize_value(text):
    """Parse a quantity string into a float, or None if not parseable.

    Handles digit groups with commas/spaces ("770,000", "300 000"), scale words
    ("5 million", "7 billion"), number words ("four"), currency prefixes
    ("US$74 million"), and percentages ("41 per cent" -> 41.0).
    """
    if not text:
        return None
    t = str(text).lower().strip()
    t = re.sub(r"(us\$|usd|\$|€|£)", "", t)
    t = re.sub(r"\s*(per\s*cent|%)\s*$", "", t)
    m = re.search(r"\d[\d,\. ]*", t)
    if m:
        num_str = m.group(0).replace(",", "").replace(" ", "").rstrip(".")
        try:
            value = float(num_str)
        except ValueError:
            return None
    else:
        value = None
        for word, v in _WORD_NUMBERS.items():
            if re.search(rf"\b{word}\b", t):
                value = float(v)
                break
        if value is None:
            return None
    for word, mult in _MULTIPLIERS.items():
        if re.search(rf"\b{word}\b", t):
            value *= mult
            break
    return value


def units_compatible(a, b):
    """Case-insensitive equality or containment ('people' ~ 'people affected')."""
    if a is None or b is None:
        return a == b
    a, b = a.lower().strip(), b.lower().strip()
    return a == b or a in b or b in a


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
# Optimal one-to-one matching (position-primary maximum-weight assignment)
# ---------------------------------------------------------------------------

def _pair_score(gt_ann: dict, pred_ann: dict):
    """Score a candidate (reference, prediction) pair.

    Position is primary: quantity-span IoU carries the largest weight.
    Normalized value equality and surface/unit similarity refine the choice
    among positional ties (repeated values, percentages). A pair is eligible
    only if the spans overlap or the values/surface forms agree; ineligible
    pairs can never be matched.
    """
    gb, ge = get_quantity_span(gt_ann)
    pb, pe = get_quantity_span(pred_ann)
    iou = 0.0
    if None not in (gb, ge, pb, pe):
        inter = max(0, min(ge, pe) - max(gb, pb))
        union = max(ge, pe) - min(gb, pb)
        iou = inter / union if union else 0.0
    gv = normalize_value(get_field_text(gt_ann, "quantity"))
    pv = normalize_value(get_field_text(pred_ann, "quantity"))
    val = 1.0 if (gv is not None and pv is not None and gv == pv) else 0.0
    text = 1.0 if get_quantity_text(gt_ann) == get_quantity_text(pred_ann) else 0.0
    unit = 1.0 if units_compatible(get_field_text(gt_ann, "unit"),
                                   get_field_text(pred_ann, "unit")) else 0.0
    eligible = iou > 0 or val > 0 or text > 0
    if not eligible:
        return None
    return 3.0 * iou + 2.0 * val + 0.5 * text + 0.25 * unit


def _hungarian_max(score):
    """Exact maximum-weight assignment (Kuhn-Munkres with potentials).

    score: rectangular matrix (rows x cols) of floats, None = ineligible.
    Returns list of (row, col) pairs restricted to eligible cells.
    Deterministic for a given input order.
    """
    n, m = len(score), len(score[0]) if score else 0
    if n == 0 or m == 0:
        return []
    size = max(n, m)
    BIG = 1e9
    # cost matrix for minimization, padded square
    a = [[BIG] * (size + 1) for _ in range(size + 1)]
    for i in range(n):
        for j in range(m):
            sc = score[i][j]
            a[i + 1][j + 1] = BIG if sc is None else -sc
    INF = float("inf")
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = a[i0][j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]
    pairs = []
    for j in range(1, size + 1):
        i = p[j]
        if 1 <= i <= n and 1 <= j <= m and score[i - 1][j - 1] is not None:
            pairs.append((i - 1, j - 1))
    return pairs


def _sort_key(ann):
    b, e = get_quantity_span(ann)
    return (b if b is not None else 10**9, e if e is not None else 10**9,
            str(get_quantity_text(ann)))


def match_annotations_optimal(gt_list: list, pred_list: list):
    """Global maximum-weight one-to-one matching over eligible pairs.

    Both lists are sorted by quantity span position before matching, so the
    procedure is deterministic and independent of file record order.
    Returns (matched_pairs, unmatched_gt, unmatched_pred).
    """
    gt_sorted = sorted(gt_list, key=_sort_key)
    pred_sorted = sorted(pred_list, key=_sort_key)
    score = [[_pair_score(g, q) for q in pred_sorted] for g in gt_sorted]
    pairs = _hungarian_max(score)
    matched = [(gt_sorted[i], pred_sorted[j]) for i, j in pairs]
    used_g = {i for i, _ in pairs}
    used_p = {j for _, j in pairs}
    unmatched_gt = [g for i, g in enumerate(gt_sorted) if i not in used_g]
    unmatched_pred = [q for j, q in enumerate(pred_sorted) if j not in used_p]
    return matched, unmatched_gt, unmatched_pred


MATCHERS = {"assignment": match_annotations_optimal, "legacy": match_annotations}


# ---------------------------------------------------------------------------
# Hierarchical attribution (semantic Shapley + localization + boundary)
# ---------------------------------------------------------------------------

SEMANTIC_CRITERIA = ["value", "eventType", "unit", "modifier"]


def hierarchical_attribution(pair_criteria: list, n_gt: int, n_pred: int) -> dict:
    """Decompose recall and precision loss into non-nested components.

    Components (each a share of the respective denominator):
      unmatched      reference/prediction had no eligible counterpart
      semantic       matched but a semantic field is wrong; split over
                     {value, eventType, unit, modifier} by exact Shapley
                     with value function v(S) = share of matched pairs
                     satisfying every criterion in S (v(empty) = 1)
      localization   semantically correct but quantity spans do not overlap
      boundary       overlapping but not character-identical spans
    The semantic Shapley uses only the four non-nested semantic criteria, so
    no coalition mixes nested span conditions; span conditions are handled
    hierarchically outside the Shapley computation.
    """
    from itertools import combinations
    from math import factorial

    matched = len(pair_criteria)
    sem_ok = [c for c in pair_criteria if all(c[x] for x in SEMANTIC_CRITERIA)]
    loc_ok = [c for c in sem_ok if c["span_overlap"]]
    full_ok = [c for c in loc_ok if c["span_exact"]]

    def shapley_matched():
        if not matched:
            return {c: 0.0 for c in SEMANTIC_CRITERIA}
        k = len(SEMANTIC_CRITERIA)
        cache = {}
        def f(subset):
            key = frozenset(subset)
            if key not in cache:
                good = sum(1 for c in pair_criteria if all(c[x] for x in key))
                cache[key] = good / matched
            return cache[key]
        shap = {}
        for crit in SEMANTIC_CRITERIA:
            others = [x for x in SEMANTIC_CRITERIA if x != crit]
            total = 0.0
            for r in range(len(others) + 1):
                for combo in combinations(others, r):
                    w = factorial(r) * factorial(k - r - 1) / factorial(k)
                    total += w * (f(combo) - f(combo + (crit,)))
            shap[crit] = total
        return shap

    shap = shapley_matched()

    def components(denom):
        if not denom:
            return {}
        m = matched / denom
        return {
            "unmatched": (denom - matched) / denom,
            "semantic": {c: shap[c] * m for c in SEMANTIC_CRITERIA},
            "semantic_total": (matched - len(sem_ok)) / denom,
            "localization": (len(sem_ok) - len(loc_ok)) / denom,
            "boundary": (len(loc_ok) - len(full_ok)) / denom,
            "surviving": len(full_ok) / denom,
        }

    return {"recall": components(n_gt), "precision": components(n_pred)}


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
# Strictness staircase — how much "agreement" survives as criteria tighten.
#
# A value-only evaluation overstates quality: two systems can agree on "5"
# while disagreeing on what the 5 counts, its type, and its bounds. The
# staircase reports the share of GT annotations that survive each cumulative
# criterion, exposing the gap a single-number score hides.
# ---------------------------------------------------------------------------

STAIRCASE_LEVELS = [
    "1_value_match",        # normalized numeric value equal
    "2_plus_eventType",     # + eventType equal
    "3_plus_unit",          # + unit compatible (containment)
    "4_plus_modifier",      # + modifier equal (absent == absent)
    "5_plus_span_overlap",  # + quantity spans overlap (provenance, relaxed)
    "6_exact_span",         # + quantity text exactly equal (provenance, strict)
]

CRITERIA = ["value", "eventType", "unit", "modifier", "span_overlap", "span_exact"]


def criteria_booleans(gt_ann: dict, pred_ann: dict) -> dict:
    """Independent (non-cumulative) pass/fail per criterion for a matched pair."""
    gt_q, pred_q = get_field_text(gt_ann, "quantity"), get_field_text(pred_ann, "quantity")
    gv, pv = normalize_value(gt_q), normalize_value(pred_q)
    gm = (get_field_text(gt_ann, "modifier") or "").lower().strip()
    pm = (get_field_text(pred_ann, "modifier") or "").lower().strip()
    gb, ge = get_quantity_span(gt_ann)
    pb, pe = get_quantity_span(pred_ann)
    overlap = (None not in (gb, ge, pb, pe)) and max(gb, pb) < min(ge, pe)
    return {
        "value": gv is not None and pv is not None and gv == pv,
        "eventType": get_field_text(gt_ann, "eventType") == get_field_text(pred_ann, "eventType"),
        "unit": units_compatible(get_field_text(gt_ann, "unit"), get_field_text(pred_ann, "unit")),
        "modifier": gm == pm,
        "span_overlap": overlap,
        "span_exact": gt_q == pred_q,
    }


def staircase_pair(gt_ann: dict, pred_ann: dict) -> dict:
    c = criteria_booleans(gt_ann, pred_ann)
    value_ok = c["value"]
    type_ok = value_ok and c["eventType"]
    unit_ok = type_ok and c["unit"]
    mod_ok = unit_ok and c["modifier"]
    ov_ok = mod_ok and c["span_overlap"]
    span_ok = ov_ok and c["span_exact"]
    return {
        "1_value_match": value_ok,
        "2_plus_eventType": type_ok,
        "3_plus_unit": unit_ok,
        "4_plus_modifier": mod_ok,
        "5_plus_span_overlap": ov_ok,
        "6_exact_span": span_ok,
    }


def shapley_attribution(pair_criteria: list, n_gt: int) -> dict:
    """Order-free attribution: Shapley share of full-strictness failures per criterion.

    The coalition payoff f(S) is the share of reference annotations whose matched
    pair satisfies every criterion in S (unmatched references satisfy nothing).
    Reported values are each criterion's Shapley contribution to the total drop
    1 - f(all criteria); they sum to that drop. This replaces reading error
    attribution off the (order-dependent) sequential staircase.
    """
    from itertools import combinations
    from math import factorial

    crits = CRITERIA
    k = len(crits)

    def payoff(subset):
        if not subset:
            return 1.0
        good = sum(1 for c in pair_criteria if all(c[x] for x in subset))
        return good / n_gt if n_gt else 0.0

    cache = {}
    def f(subset):
        key = frozenset(subset)
        if key not in cache:
            cache[key] = payoff(key)
        return cache[key]

    shap = {}
    for crit in crits:
        others = [x for x in crits if x != crit]
        total = 0.0
        for r in range(len(others) + 1):
            for combo in combinations(others, r):
                w = factorial(r) * factorial(k - r - 1) / factorial(k)
                total += w * (f(combo) - f(combo + (crit,)))
        shap[crit] = total
    return shap


def staircase_document(matched_pairs: list, n_gt: int, n_pred: int = None) -> dict:
    """Counts, recall, precision, and F1 per cumulative level, plus per-pair
    criteria booleans for order-free attribution."""
    counts = {lvl: 0 for lvl in STAIRCASE_LEVELS}
    pair_criteria = []
    for gt_ann, pred_ann in matched_pairs:
        result = staircase_pair(gt_ann, pred_ann)
        pair_criteria.append(criteria_booleans(gt_ann, pred_ann))
        for lvl in STAIRCASE_LEVELS:
            counts[lvl] += result[lvl]
    if n_pred is None:
        n_pred = len(matched_pairs)
    out = {"n_gt": n_gt, "n_pred": n_pred, "counts": counts,
           "criteria": pair_criteria,
           "recall": {}, "precision": {}, "f1": {},
           "shares": {}}
    for lvl in STAIRCASE_LEVELS:
        c = counts[lvl]
        r = c / n_gt if n_gt else 0.0
        p = c / n_pred if n_pred else 0.0
        out["recall"][lvl] = r
        out["precision"][lvl] = p
        out["f1"][lvl] = 2 * p * r / (p + r) if (p + r) else 0.0
        out["shares"][lvl] = r  # kept for backward compatibility
    return out


# ---------------------------------------------------------------------------
# Per-document evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(gt_list: list, pred_list: list, matcher: str = "assignment") -> dict:
    """Returns {field: FieldAccumulator} for one document."""
    accs = {f: FieldAccumulator() for f in ALL_FIELDS}

    matched_pairs, unmatched_gt, unmatched_pred = MATCHERS[matcher](gt_list, pred_list)

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


def save_csv(out_path, doc_results, global_accs, macro_doc,
             staircase_results=None, global_staircase_counts=None, global_staircase_gt=0,
             global_staircase_pred=0, global_criteria=None):
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
        if staircase_results:
            for stem, stair in staircase_results:
                for lvl in STAIRCASE_LEVELS:
                    w.writerow(["staircase", stem, lvl,
                                stair["precision"][lvl], stair["recall"][lvl],
                                stair["f1"][lvl], "",
                                stair["counts"][lvl], stair["n_pred"], stair["n_gt"]])
            for lvl in STAIRCASE_LEVELS:
                c = global_staircase_counts[lvl]
                r = c / global_staircase_gt if global_staircase_gt else 0.0
                pr = c / global_staircase_pred if global_staircase_pred else 0.0
                f1 = 2 * pr * r / (pr + r) if (pr + r) else 0.0
                w.writerow(["staircase", "ALL", lvl, pr, r, f1, "",
                            c, global_staircase_pred, global_staircase_gt])
            shap = shapley_attribution(global_criteria, global_staircase_gt)
            for crit, drop in shap.items():
                w.writerow(["shapley", "ALL", crit, drop, "", "", "", "", "", ""])
    print(f"\nResults saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",   required=True, help="Ground truth directory")
    parser.add_argument("--pred", required=True, help="Predictions directory")
    parser.add_argument("--out",  default=None,  help="Optional CSV output path")
    parser.add_argument("--matcher", choices=["assignment", "legacy"], default="assignment",
                        help="assignment: position-primary maximum-weight one-to-one "
                             "matching (default); legacy: surface-form-first greedy")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Number of document-level bootstrap resamples for "
                             "staircase confidence intervals (0 = off)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = discover_pairs(args.gt, args.pred)
    if not pairs:
        print("No matching file pairs found.")
        sys.exit(1)
    print(f"Found {len(pairs)} matching document pair(s).")

    doc_results = []
    global_accs = {f: GlobalAccumulator() for f in ALL_FIELDS}
    staircase_results = []
    global_staircase_counts = {lvl: 0 for lvl in STAIRCASE_LEVELS}
    global_staircase_gt = 0
    global_staircase_pred = 0
    global_criteria = []

    for stem, gt_path, pred_path in pairs:
        gt_list   = load_annotations(gt_path)
        pred_list = load_annotations(pred_path)
        accs      = evaluate_pair(gt_list, pred_list, args.matcher)
        doc_results.append((stem, accs))
        for field, acc in accs.items():
            global_accs[field].merge(acc)

        matched_pairs, _, _ = MATCHERS[args.matcher](gt_list, pred_list)
        stair = staircase_document(matched_pairs, len(gt_list), len(pred_list))
        staircase_results.append((stem, stair))
        for lvl in STAIRCASE_LEVELS:
            global_staircase_counts[lvl] += stair["counts"][lvl]
        global_staircase_gt += stair["n_gt"]
        global_staircase_pred += stair["n_pred"]
        global_criteria.extend(stair["criteria"])

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

    # Strictness staircase: precision, recall, F1 at every cumulative level
    print(f"\n{'='*76}")
    print("  STRICTNESS STAIRCASE (cumulative criteria; P/R/F1 over all annotations)")
    print(f"{'='*76}")
    print(f"{'Level':<24}{'Count':>8}{'Recall':>10}{'Precision':>11}{'F1':>10}")
    print("-" * 64)
    for lvl in STAIRCASE_LEVELS:
        count = global_staircase_counts[lvl]
        r = count / global_staircase_gt if global_staircase_gt else 0.0
        pr = count / global_staircase_pred if global_staircase_pred else 0.0
        f1 = 2 * pr * r / (pr + r) if (pr + r) else 0.0
        print(f"{lvl:<24}{count:>8}{r:>10.1%}{pr:>11.1%}{f1:>10.3f}")
    print(f"{'(GT annotations)':<24}{global_staircase_gt:>8}")
    print(f"{'(predicted annotations)':<24}{global_staircase_pred:>8}")

    # Hierarchical attribution (semantic Shapley + localization + boundary)
    attr = hierarchical_attribution(global_criteria, global_staircase_gt, global_staircase_pred)
    print(f"\n{'='*76}")
    print("  HIERARCHICAL ATTRIBUTION (share of loss; semantic split by Shapley)")
    print(f"{'='*76}")
    print(f"{'Component':<22}{'Recall':>10}{'Precision':>12}")
    print("-" * 46)
    r, pr = attr["recall"], attr["precision"]
    print(f"{'unmatched':<22}{r['unmatched']:>10.1%}{pr['unmatched']:>12.1%}")
    for c in SEMANTIC_CRITERIA:
        print(f"{'semantic: ' + c:<22}{r['semantic'][c]:>10.1%}{pr['semantic'][c]:>12.1%}")
    print(f"{'localization':<22}{r['localization']:>10.1%}{pr['localization']:>12.1%}")
    print(f"{'boundary (exact span)':<22}{r['boundary']:>10.1%}{pr['boundary']:>12.1%}")
    print(f"{'(surviving, full)':<22}{r['surviving']:>10.1%}{pr['surviving']:>12.1%}")

    # Document-level bootstrap confidence intervals
    if args.bootstrap:
        import random
        rng = random.Random(args.seed)
        docs = [(st["counts"], st["n_gt"], st["n_pred"]) for _, st in staircase_results]
        lvls = ["1_value_match", STAIRCASE_LEVELS[-1]]
        samples = {lvl: [] for lvl in lvls}
        for _ in range(args.bootstrap):
            picks = [docs[rng.randrange(len(docs))] for _ in docs]
            for lvl in lvls:
                c = sum(d[0][lvl] for d in picks)
                g = sum(d[1] for d in picks)
                q = sum(d[2] for d in picks)
                rr = c / g if g else 0.0
                pp = c / q if q else 0.0
                samples[lvl].append(2 * pp * rr / (pp + rr) if (pp + rr) else 0.0)
        print(f"\n{'='*76}")
        print(f"  BOOTSTRAP 95% CI (F1, {args.bootstrap} document resamples, seed {args.seed})")
        print(f"{'='*76}")
        for lvl in lvls:
            xs = sorted(samples[lvl])
            lo = xs[int(0.025 * len(xs))]
            hi = xs[int(0.975 * len(xs)) - 1]
            print(f"{lvl:<24}[{lo:.3f}, {hi:.3f}]")

    if args.out:
        save_csv(args.out, doc_results, global_accs, macro_doc,
                 staircase_results, global_staircase_counts, global_staircase_gt,
                 global_staircase_pred, global_criteria)


if __name__ == "__main__":
    main()