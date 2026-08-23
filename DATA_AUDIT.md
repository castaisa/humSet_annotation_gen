# Ground-truth count reconciliation (open issue)

Audit date: 23 August 2026. The counts in the distributed `Data/GroundTruthISI/`
JSON files do not match the published description of the quantitative layer
(Liberatore et al., GoodIT'24; reproduced in the paper's dataset section).
This must be reconciled before submission, since evaluation denominators
depend on it.

## What is on disk (`Data/GroundTruthISI/`, parsed JSON)

| Measure | Count |
|---|---|
| Files | 775 |
| Files with zero annotations | 27 |
| Files with >= 1 annotation | 748 |
| Annotation events (all have a quantity) | 5,262 |
| ... with unit | 4,756 |
| ... with eventType / eventDescription | 4,492 |
| ... with modifier | 591 |

## What the published description says

| Measure | Count |
|---|---|
| Excerpts sampled | 780 |
| Excerpts with >= 1 annotation | 755 |
| Number spans | 4,352 |
| Unit spans | 3,011 |
| Modifier spans | 461 |
| EventP + EventA + EventO spans | 812 + 437 + 1,244 = 2,493 |

## Discrepancies to resolve

1. 775 distributed files vs 780 sampled; 748 non-empty vs 755 reported.
2. 5,262 annotation events vs 4,352 Number spans (+910).
3. Unit (4,756 vs 3,011), modifier (591 vs 461), and event labels
   (4,492 vs 2,493) all higher in the parsed JSON.

## Likely causes (to verify against the original XMI)

- `src/parser.py` may expand compound or multi-span annotations into
  multiple events during XMI-to-JSON conversion.
- The paper counted labelled spans, whereas the JSON counts events in which
  a field is present; a unit span shared by two numbers would count once as
  a span but twice as a field.
- Possible release-version differences between the annotated XMI batches.

## Action

Re-derive all counts from the original XMI with a documented script and state
in the paper which representation the evaluation uses (the parsed event-level
JSON, n = 5,262). Until then, evaluation results should cite the on-disk
counts above, not the published span counts.
