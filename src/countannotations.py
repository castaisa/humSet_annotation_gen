#Contamos la cantidad de annotations que tiene GroundTruthISI


import json
import os
from pathlib import Path

def count_annotations(folder_path):
    """Count the number of JSON documents in GroundTruthISI folder"""
    json_files = list(Path(folder_path).glob('*.json'))
    file_count = len(json_files)
    print(f"Total JSON documents: {file_count}")
    return file_count

if __name__ == "__main__":
    count_annotations("/humSet_annotation_gen/Data/GroundTruthISI")