import xml.etree.ElementTree as ET
import json
import os
from pathlib import Path

# Carpetas
input_base_dir = "annotationsSinParsear"  # Carpeta base con las ~770 carpetas
output_folder = "GroundTruthISI"         # Carpeta donde se guardarán los .json
os.makedirs(output_folder, exist_ok=True)

# Namespace map to handle prefixed tags in the XMI
namespaces = {
    'cas': 'http:///uima/cas.ecore',
    'custom': 'http:///custom.ecore',
    'xmi': 'http://www.omg.org/XMI'
}

# Categories to extract and how they map to JSON fields
categories = {
    "Number": "quantity",
    "Unit": "unit",
    "Modifier": "modifier",
    "EventP": "eventDescription",
    "EventA": "eventDescription",
    "EventO": "eventDescription"
}

# Function to extract annotator number from folder name
def get_annotator_number(folder_name):
    """
    Extrae el número del anotador del nombre de la carpeta.
    Por ejemplo: 'a1_1234332' -> 'annotator1'
    """
    if folder_name.startswith('a') and '_' in folder_name:
        annotator_num = folder_name[1]
        if annotator_num.isdigit():
            return f"annotator{annotator_num}"
    return None

# Function to find XMI file in annotator folder
def find_xmi_file(annotator_path):
    """Busca el archivo .xmi en la carpeta del anotador."""
    for file in os.listdir(annotator_path):
        if file.endswith('.xmi'):
            return os.path.join(annotator_path, file)
    return None

# Contadores
success_count = 0
error_count = 0
skipped_count = 0

# Get all subdirectories in annotationSinParsear
all_folders = [f for f in os.listdir(input_base_dir) 
               if os.path.isdir(os.path.join(input_base_dir, f))]

print(f"Encontradas {len(all_folders)} carpetas para procesar\n")

# Process each folder
for folder_name in sorted(all_folders):
    folder_path = os.path.join(input_base_dir, folder_name)
    
    # Determine which annotator folder to use
    annotator_folder = get_annotator_number(folder_name)
    
    if annotator_folder is None:
        print(f"⊘ Saltado: {folder_name} (no se pudo determinar el número de anotador)")
        skipped_count += 1
        continue
    
    # Build path to annotator folder
    annotator_path = os.path.join(folder_path, annotator_folder)
    
    if not os.path.exists(annotator_path):
        print(f"⊘ Saltado: {folder_name} (no existe {annotator_folder})")
        skipped_count += 1
        continue
    
    # Find the XMI file
    input_xmi_file = find_xmi_file(annotator_path)
    
    if input_xmi_file is None:
        print(f"⊘ Saltado: {folder_name} (no se encontró archivo .xmi en {annotator_folder})")
        skipped_count += 1
        continue
    
    # Create output filename
    output_json_file = os.path.join(output_folder, f"{folder_name}.json")
    
    # # If file already exists, skip processing (safety feature)
    # if os.path.exists(output_json_file):
    #     print(f"⊘ Saltado: {folder_name} - JSON ya existe.")
    #     skipped_count += 1
    #     continue
    
    try:
        tree = ET.parse(input_xmi_file)
        root = tree.getroot()

        def extract_spans():
            spans = {}
            for span in root.findall('.//custom:Span', namespaces):
                span_id = span.get('{http://www.omg.org/XMI}id')
                label = span.get('label')
                if label in categories:
                    begin = int(span.get('begin'))
                    end = int(span.get('end'))
                    text = root.find('.//cas:Sofa', namespaces).get('sofaString')[begin:end]
                    spans[span_id] = (label, text.strip(), begin, end)
            return spans

        def extract_relations():
            relations = []
            for relation in root.findall('.//custom:Relation', namespaces):
                governor = relation.get('Governor')
                dependent = relation.get('Dependent')
                relations.append((governor, dependent))
            return relations

        def build_json_data(spans, relations):
            events = []
            event_map = {}

            for governor, dependent in relations:
                if governor not in spans or dependent not in spans:
                    continue
                if governor not in event_map:
                    event_map[governor] = {}
                event_map[governor][dependent] = spans[dependent]

            for governor, dependents in event_map.items():
                governor_label, governor_text, governor_begin, governor_end = spans[governor]
                event = {
                    categories[governor_label]: {
                        "text": governor_text,
                        "begin": governor_begin,
                        "end": governor_end
                    }
                }

                for dependent, (label, text, begin, end) in dependents.items():
                    key = categories[label]
                    event[key] = {
                        "text": text,
                        "begin": begin,
                        "end": end
                    }
                    if label.startswith("Event"):
                        event["eventType"] = label

                if "quantity" in event:
                    events.append(event)

            return events

        spans = extract_spans()
        relations = extract_relations()
        parsed_data = build_json_data(spans, relations)

        with open(output_json_file, 'w', encoding='utf-8') as json_file:
            json.dump(parsed_data, json_file, indent=2, ensure_ascii=False)

        print(f"✓ Procesado: {folder_name} -> {os.path.basename(output_json_file)}")
        success_count += 1

    except Exception as e:
        print(f"✗ Error procesando {folder_name}: {str(e)}")
        error_count += 1

print(f"\n{'='*60}")
print(f"Procesamiento completado:")
print(f"  ✓ Exitosos: {success_count}")
print(f"  ✗ Errores: {error_count}")
print(f"  ⊘ Saltados: {skipped_count}")
print(f"  Total carpetas: {len(all_folders)}")
print(f"{'='*60}")