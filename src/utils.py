import os
import json


def analyze_nulls_all_fields(annotation_dir):
    fields = ["eventDescription", "modifier", "quantity", "unit"]

    total_documents = 0
    total_annotations = 0

    stats = {
        field: {
            "total_annotations_with_field": 0,
            "null_annotations": 0,
            "documents_with_at_least_one_null": 0
        }
        for field in fields
    }

    for file in os.listdir(annotation_dir):
        if not file.endswith(".json"):
            continue

        total_documents += 1
        path = os.path.join(annotation_dir, file)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        doc_has_null_for_field = {field: False for field in fields}

        for event in data:
            total_annotations += 1

            for field in fields:
                if field not in event:
                    continue

                value = event[field]
                stats[field]["total_annotations_with_field"] += 1

                if value.get("start") is None or value.get("end") is None:
                    stats[field]["null_annotations"] += 1
                    doc_has_null_for_field[field] = True

        for field in fields:
            if doc_has_null_for_field[field]:
                stats[field]["documents_with_at_least_one_null"] += 1

    print(f"\n=== NULL ANALYSIS FOR: {annotation_dir} ===\n")
    print("Total documents:", total_documents)
    print("Total annotations:", total_annotations)
    print()

    for field in fields:
        total_with_field = stats[field]["total_annotations_with_field"]
        null_annotations = stats[field]["null_annotations"]
        docs_with_null = stats[field]["documents_with_at_least_one_null"]

        annotation_null_pct = 0
        if total_with_field > 0:
            annotation_null_pct = (null_annotations / total_with_field) * 100

        document_null_pct = 0
        if total_documents > 0:
            document_null_pct = (docs_with_null / total_documents) * 100

        print(f"--- Field: {field} ---")
        print("Annotations with this field:", total_with_field)
        print("Null annotations:", null_annotations)
        print("Null annotation %:", round(annotation_null_pct, 2), "%")
        print("Documents with at least one null:", docs_with_null)
        print("Document % with at least one null:", round(document_null_pct, 2), "%")
        print()