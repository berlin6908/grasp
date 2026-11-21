---
name: grasp-evaluation
description: Evaluate saved Grasp Final predictions and cited evidence, verify S/I metrics, or analyze saved support-removal outcomes using the repository evaluators. Applies to CWQ, WebQSP and 2WikiMultiHopQA, without model inference.
---

# Grasp evaluation

Read [docs/evaluation.md](../../docs/evaluation.md) for the metric contract
and [docs/data.md](../../docs/data.md) for benchmark inputs. The supplied
evaluator consumes Grasp Final records. Baseline results are bundled for
analysis, but their inference code and native evaluation adapters are not
part of this release.

## Evaluate saved predictions

Use `bash scripts/evaluate.sh` or `python -m eva.evaluate`; inspect `--help`
for the selected benchmark. Run from the repository root. Reuse the existing
answer matcher, citation parser and replay implementation rather than writing
a second scoring implementation.

| Concern | Existing implementation |
|---|---|
| Final adaptation and population evaluation | `eva/evaluate.py` |
| Answer matching and aggregation | `eva/metrics.py` |
| Cited evidence resolution | `eva/evidence.py` |
| Benchmark-specific support replay | `eva/replay.py` and `dataset_split/` |
| Primary and progressive transitions | `eva/cwq_rq2_outcomes.py`, `eva/cwq_progressive_outcomes.py` |

Supply the registered dataset, Final file(s), partition index and benchmark
annotations. For a curated subset, select matching Final IDs first: extra
Final IDs are invalid, while missing predictions remain in the registered
denominator. Decompress `.jsonl.gz` before passing it to the JSONL CLI.
Use `--flat-evidence` for the component without specification-aware answering.
Write new evaluation outputs to a separate run directory.

CWQ and WebQSP replay the credited answer's constraints over cited evidence
and the registered background facts. 2WikiMultiHopQA checks every required
fact using the frozen endpoint and relation rules. Returned evidence is
distinct from all retrieved or packed triples. Evaluate registered rows
uniformly, then aggregate with the supplied partition labels; X stays in
the full diagnostics and is excluded from S/I paper measures.

Use the six paper measures defined in `docs/evaluation.md`. Errors, missing
predictions and invalid evidence keep their fixed denominators and receive
no Unknown credit. EvidenceGap is undefined when there are no correct S
predictions. EBA combines integer counts over S + I, not two equally weighted
partition rates. Preserve internal diagnostic field names when inspecting
old records; do not relabel their full-population rates as paper S rates.

## Support removal

Follow [docs/interventions.md](../../docs/interventions.md) for normalization
and scoring. The primary and progressive tools use the registered paper
cohorts. A different population is a different experiment, not an override
of the fixed denominator.

Pair with the selected condition's recorded original reference. In the
supplementary prompt comparisons this is the same arm's G+ or progressive
k=0, not its separate natural-input output. Keep Same, Other, Unknown and
Error counts distinct; APR is the Same rate. Preserve recorded reviewed
display views separately from canonical evaluator outcomes.

## Verify the requested result

Check the selected IDs and counts, retained errors and appropriate evidence
interface, then compare the derived rates to the written summary. Do not
claim missing evidence metrics for an answer-only interface. For a new local
setup, `bash scripts/smoke.sh runs/evaluation_smoke` exercises split building
and evaluation on synthetic saved answers without model calls. It does not
validate real-model performance.
