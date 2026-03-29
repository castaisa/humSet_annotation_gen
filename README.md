# humSet_annotation_gen

Annotation generation, evaluation, and offset processing for HumSet humanitarian data.

## Overview

This repository contains:
- Original HumSet text chunks
- Data corrected using external tools
- Annotation generation with configurable LLM models
- Metrics evaluation
- Dynamic offset alignment using two-stage processing

## Architecture

**Two-Stage Offset Processing:**

1. **offsetFinderQuant.py** — Uses dynamic programming to align quantity field annotations to character spans in text. This establishes precise offsets for numerical values.

2. **offsetFinderRest.py** — Uses the quantity offsets as anchors to calculate character spans for remaining fields (modifier, unit, eventDescription, eventType) relative to the quantity positions.

## Structure

### `src/` — Scripts
- **annotationsGen.py**: Generate annotations from text using an LLM model (GPT-4, GPT-4 mini, etc.).
- **metricsCalc.py**: Evaluate predictions against ground truth (precision, recall, F1, Levenshtein).
- **offsetFinderQuant.py**: Dynamic programming alignment for quantity field offsets.
- **offsetFinderRest.py**: Align remaining fields using quantity offsets as reference.
- **parser.py**: Annotation data parser from .xmi to .json.
- **utils.py**: Helper functions.

### `Data/` — Datasets
- **text_sources/**: Original HumSet text file chunks (.txt).
- **GroundTruthISI/**: Original ISI annotations (.json).
- **annotationsGPT4.1/**: Generated annotations (.json).
- **annotations_with_offsets_dinamic/**: Annotations with offsets obtained with a dynamic programing algorithm.
- **annotations_with_offsets_quantbased/**: Final processed annotations with both quantity and field offsets.
- **annotationsWithoutParse/**: Raw annotations before parsing.
- **results.csv**: Evaluation metrics.

## Quick Start

```bash
# Generate annotations 
python src/annotationsGen.py

# Calculate quantity offsets (dynamic programming)
python src/offsetFinderQuant.py

# Calculate remaining field offsets
python src/offsetFinderRest.py

# Evaluate metrics
python src/metricsCalc.py --gt Data/GroundTruthISI --pred Data/annotationsGPT4.1 --out Data/results.csv