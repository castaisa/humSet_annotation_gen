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

---

## RESOLVED — programmatic reconciliation against the original XMI (23 Aug 2026)

The full XMI-to-JSON audit was run over `Data/annotationsWithoutParse/`
(778 document folders, one XMI per assigned annotator, replicating
`src/parser.py` logic exactly). Results:

| Stage | Count |
|---|---|
| Document folders | 778 |
| Skipped (no XMI for assigned annotator) | 3 → explains 775 JSON files |
| XMI Number spans | 5,329 |
| XMI Unit / Modifier spans | 3,774 / 550 |
| XMI Event spans (P/A/O) | 952 / 517 / 1,535 |
| Relations | 10,286 (all valid) |
| JSON records built | 5,262 |
| Records from Number governors | 5,262 (100%) |
| Records from Event governors (duplication path) | **0** |
| Numbers appearing in more than one record | **0** |
| Numbers with no relation (produce no record, dropped) | 63 (1.2%) |
| Records with >1 Unit dependent (silent overwrite) | 176 (3.3%) |
| Records with >1 Event dependent (eventType overwritten) | 250 (4.8%) |

**Conclusions.**
1. No combinatorial expansion and no duplicate records: each JSON record
   corresponds to exactly one annotated Number span with at least one
   relation. Records are one-per-number and statistically independent at
   the same level as the underlying annotations.
2. The 5,262 vs 4,352 gap is release growth, not construction artefact:
   the distributed XMI batches contain 5,329 Number spans, more than the
   snapshot counted in the GoodIT'24 paper description.
3. Real conversion losses are small and now quantified: 63 unrelated
   numbers are dropped, and multi-Unit (176) and multi-Event (250)
   relations are silently collapsed by dict overwrite, which makes the
   retained unit/eventType dependent on XML iteration order for those
   records. Fixing the parser to keep all dependents (or the first by
   span order, deterministically) is recommended before the next release.
4. The paper should cite: 775 files, 748 non-empty, 5,262 records, one
   record per related Number span, with the three loss modes above
   disclosed.
