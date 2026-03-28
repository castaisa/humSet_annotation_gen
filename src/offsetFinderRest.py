import os
import json
import re
from src.utils import analyze_nulls_all_fields

def find_all_spans(text, target):
    spans = []
    for m in re.finditer(re.escape(target), text, flags=re.IGNORECASE):
        spans.append((m.start(), m.end()))
    return spans


def choose_closest_span(spans, q_start, q_end):
    if not spans:
        return None

    best_span = None
    best_dist = float("inf")

    for start, end in spans:
        if end <= q_start:
            dist = q_start - end
        elif start >= q_end:
            dist = start - q_end
        else:
            dist = 0

        if dist < best_dist:
            best_dist = dist
            best_span = (start, end)

    return best_span


def find_closest_span_windowed(text, target, q_start, q_end, window=300):
    left = max(0, q_start - window)
    right = min(len(text), q_end + window)

    local_text = text[left:right]
    local_spans = []

    for m in re.finditer(re.escape(target), local_text, flags=re.IGNORECASE):
        start = left + m.start()
        end = left + m.end()
        local_spans.append((start, end))

    best = choose_closest_span(local_spans, q_start, q_end)
    if best is not None:
        return {
            "text": target,
            "start": best[0],
            "end": best[1]
        }

    global_spans = find_all_spans(text, target)
    best = choose_closest_span(global_spans, q_start, q_end)

    if best is not None:
        return {
            "text": target,
            "start": best[0],
            "end": best[1]
        }

    return {
        "text": target,
        "start": None,
        "end": None
    }


def realign_non_quantity_fields(text, annotations, window=300):
    fields_to_realign = ["eventDescription", "unit", "modifier"]
    results = []

    for ann in annotations:
        new_event = {}

        if "eventType" in ann:
            new_event["eventType"] = ann["eventType"]

        if "quantity" in ann:
            new_event["quantity"] = ann["quantity"]

        quantity = ann.get("quantity")

        for field in fields_to_realign:
            if field not in ann or not ann[field]:
                continue

            value = ann[field]["text"] if isinstance(ann[field], dict) else ann[field]

            if quantity and quantity.get("start") is not None and quantity.get("end") is not None:
                new_event[field] = find_closest_span_windowed(
                    text=text,
                    target=value,
                    q_start=quantity["start"],
                    q_end=quantity["end"],
                    window=window
                )
            else:
                new_event[field] = {
                    "text": value,
                    "start": None,
                    "end": None
                }

        results.append(new_event)

    return results


def process_existing_offsets(source_dir, existing_offsets_dir, output_dir, window=300):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(existing_offsets_dir):
        if not file.endswith(".json"):
            continue

        base = os.path.splitext(file)[0]
        text_path = os.path.join(source_dir, base + ".txt")
        ann_path = os.path.join(existing_offsets_dir, file)

        if not os.path.exists(text_path):
            print("Missing source text:", base + ".txt")
            continue

        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        with open(ann_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        results = realign_non_quantity_fields(text, annotations, window=window)

        out_path = os.path.join(output_dir, file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        #print("Processed", file)


process_existing_offsets(
    source_dir="text_sources",
    existing_offsets_dir="annotations_with_offsets_dinamic",
    output_dir="annotations_with_offsets_quantbased",
    window=300
)

analyze_nulls_all_fields("annotations_with_offsets_quantbased")