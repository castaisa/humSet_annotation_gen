import os
import json
import re
from typing import List, Tuple, Dict
from functools import lru_cache
from src.utils import analyze_nulls_all_fields


def result_parse(text: str, entities: List[str]) -> Tuple[List[Dict[str, int]], int]:
    """Align detected entity strings to their character spans in the original text.

    This function receives the original `text` and an ordered list of entity
    strings (`entities`). It attempts to map each entity, in order, to a non-
    overlapping occurrence in `text` by finding character start/end indices.

    It uses dynamic programming to explore two choices for each entity:
      1) Match it at the next valid occurrence after the current index.
      2) Skip it (counted as one error) if no suitable occurrence should be used.

    The DP minimizes the number of skipped entities (errors). For matched
    entities, the function returns dictionaries with keys: `text`, `start`,
    and `end`. For skipped ones, only `text` is returned. Consumers typically
    filter out the skipped entries later.

    Returns a tuple of:
      - a list of entity dicts (some with spans, some without when skipped)
      - the total number of skipped entities (minimal under the constraints)
    """

    @lru_cache(None)
    def dp(i: int, idx: int) -> Tuple[int, List[Dict[str, int]]]:
        # Base case: processed all entities — no further errors, no more spans
        if i == len(entities):
            return 0, []

        entity = entities[i]
        min_errors = float('inf')
        best_result = []

        # Try to match current entity at every valid occurrence from `idx` onward
        start_idx = idx
        while True:
            # Word-boundary-like match: avoid partial matches within words
            m = re.search(rf"(?<![A-Za-z]){re.escape(entity)}(?![A-Za-z])", text[start_idx:])
            start = m.start() + start_idx if m else -1
            if start == -1:
                break

            end = start + len(entity)
            # Recurse to place the next entities after this span
            next_errors, next_result = dp(i + 1, end)

            if next_errors < min_errors:
                min_errors = next_errors
                best_result = [{'text': entity, 'start': start, 'end': end}] + next_result

            # Continue searching for a later occurrence to see if it yields fewer skips
            start_idx = start + 1

        # Option: skip this entity (incurs one error) and keep looking from the same index
        next_errors, next_result = dp(i + 1, idx)
        next_errors += 1

        if next_errors < min_errors:
            min_errors = next_errors
            best_result = [{'text': entity}] + next_result

        return min_errors, best_result

    # Start DP from the first entity and the beginning of the text
    error_count, result = dp(0, 0)
    return result, error_count


def align_event(text, annotations):

    field_types = ["eventDescription", "modifier", "quantity", "unit"]

    # 1. recolectar todos los valores por tipo
    collected = {f: [] for f in field_types}

    for ann in annotations:
        for f in field_types:
            if f in ann and ann[f]:
                collected[f].append(ann[f])

    # 2. correr result_parse por tipo
    aligned = {}

    for f in field_types:
        if collected[f]:
            spans, _ = result_parse(text, collected[f])
            aligned[f] = spans
        else:
            aligned[f] = []

    # 3. reconstruir eventos
    results = []
    n_events = len(annotations)

    counters = {f: 0 for f in field_types}

    for i in range(n_events):

        ann = annotations[i]
        event = {"eventType": ann.get("eventType")}

        for f in field_types:

            if f in ann and ann[f]:

                span = aligned[f][counters[f]]
                counters[f] += 1

                if "start" in span:
                    event[f] = span
                else:
                    event[f] = {
                        "text": ann[f],
                        "start": None,
                        "end": None
                    }

        results.append(event)

    return results


def process_dataset(source_dir, annotation_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(source_dir):

        if not file.endswith(".txt"):
            continue

        base = os.path.splitext(file)[0]

        text_path = os.path.join(source_dir, file)
        ann_path = os.path.join(annotation_dir, base + ".json")

        if not os.path.exists(ann_path):
            print("Missing annotation:", file)
            continue

        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        with open(ann_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        results = []

        results = align_event(text, annotations)

        out_path = os.path.join(output_dir, base + ".json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("Processed", file)


analyze_nulls_all_fields("annotations_with_offsets_dinamic")