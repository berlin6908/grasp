---
name: grasp-results-report
description: Find, summarize and compare existing Grasp KGQA experiment results across CWQ, WebQSP and 2WikiMultiHopQA, including prompt controls and specification-quality audits. Use for result or completion questions without launching inference.
---

# Grasp results report

Resolve the requested system, dataset, model and condition in
[artifact/index.json](../../artifact/index.json). Use
[artifact/README.md](../../artifact/README.md) for the original bundle and
[supplement/README.md](../../artifact/supplement/README.md) for the completed
supplement. All commands and artifact paths refer to the repository root.

## Find the evidence

- `artifact/paper/` contains selected original table cells and figure inputs.
  The associated `index.json` run identifies available evaluation or Final
  records. A summary-only condition does not establish availability of raw
  predictions.
- `artifact/supplement/natural_metrics.csv` indexes supplementary natural
  comparisons. Each row's `artifact` identifies its `metrics.json`, selected
  `details.jsonl.gz`, and canonical `source_summary.json`. Use `metrics.json`
  for the selected paper S/I view; source summaries can include larger original
  populations and full-population diagnostics.
- Supplementary primary and progressive transitions are in
  `artifact/supplement/support_removal.csv` and `progressive.csv`. Their
  `artifact` directories contain the paired records and source summaries.
- `artifact/supplement/specification_audit/` contains the raw audit, joined
  downstream records, summary and confidence reports. Read its protocol
  before interpreting verdicts or confidence.
- `artifact/sources.tsv` records the selections and source identifiers.
  Source identifiers are provenance labels, not file paths to open.

## Report the selected condition

Identify the answer model separately from a path model or grounding scorer.
For GCR*, the Qwen path model and GPT-OSS answer model have different roles;
for Grasp, a model label can identify the scorer while planning and answering
use GPT-OSS. Preserve the original/Default/Unknown/reused-reference labels.
In particular, the matched CoT Default rerun and its historical main-table
row are different recorded conditions.

Use [docs/evaluation.md](../../docs/evaluation.md) for formulas. Report
Hit@1, EvidenceHit@1, EvidenceGap, CAR_I, OAR_S and EBA with their counts and
denominators. Read the units: JSON rates are fractions, whereas the selected
paper and supplementary natural display CSVs contain percentages. Do not
convert an absolute full-population gap into the relative EvidenceGap.

For interventions, pair critical/sham outcomes with the same arm's G+;
progressive outcomes use that arm's k=0. The separate natural-input prediction
is not the reference. Report Error separately from Unknown and distinguish
canonical outcomes from an explicitly documented reviewed paper display.
Read the cohort denominator from the selected artifact rather than reusing
a historical count.

The audit is model-assisted, keeps correct/incorrect/uncertain verdicts, and
uses no selected confidence cutoff. High confidence can accompany an
incorrect verdict. Downstream strata have their own S/I denominators and show
association, not an effect of repairing plans.

Lead with the requested results, then the condition and source links needed
to interpret them. Keep unavailable metrics blank or marked unavailable.
Use the index's unfinished-condition notes for the published snapshot;
they do not establish live remote-job status. Reporting existing results
does not require model execution, rescoring or publication.
