---
name: grasp-paper-figures
description: Reproduce or revise Grasp paper and supplementary tables and statistical figures from local-results.ipynb and bundled artifacts. Use for RQ1 evidence gaps, RQ2 removal, component analyses, relation relaxation and specification-quality plots.
---

# Grasp paper figures

Use [local-results.ipynb](../../local-results.ipynb) as the maintained plotting
entrypoint. It reads only bundled artifacts and exports CSV, PDF and PNG to
`runs/paper-results/`. [artifact/README.md](../../artifact/README.md) and
[supplement/README.md](../../artifact/supplement/README.md) define available
conditions and their reporting conventions. Run from the repository root.

## Choose the data view

| Figure or table | Source |
|---|---|
| Original natural-input paper tables | `artifact/paper/main_tables.csv` |
| Original constraint-count/structure panels | `artifact/paper/rq1_gap_data.json` |
| Original primary and progressive interventions | `artifact/paper/support_removal_table.csv`, `artifact/paper/rq2_figure_data.csv` |
| Components and scorer size | `artifact/paper/components.csv`, `artifact/paper/scorer_size.csv` |
| Original semantic and partition relaxation | `artifact/analysis/` |
| Supplementary natural, primary and progressive comparisons | The three table CSVs in `artifact/supplement/` |
| Current-condition structure and relation relaxation | `artifact/supplement/diagnostics/` |
| Specification quality and downstream strata | `artifact/supplement/specification_audit/` |

Match the requested condition before editing. Original paper-selected results
and later matched prompt controls have separate protocols. Do not replace
a missing requested result with a different model or reporting view.

EvidenceGap is relative to correct S predictions, not the whole dataset.
Keep query-size and structure memberships from the frozen figure data.
Relation-relaxation curves fix the original S population and predictions;
partition-relaxation analysis instead recomputes input-graph membership.
Use the appropriate counts for each plot.

APR uses Same divided by the registered intervention denominator. Keep Error
separate in canonical supplementary plots. If a selected original paper table
combines Other and Error, preserve and name that display convention. The
supplement's Original and Unknown diagnostic panels include the same Grasp
Full reference; CoT/DoG in those panels have different answer models, so these
curves do not establish an isolated prompt effect.

## Edit, execute and inspect

Reuse the notebook's data loaders, styles and `save_figure`/`show_table` helpers.
Keep tables derived from the selected artifacts, not copied numeric constants.
Retain model labels, units and denominator information that make comparisons
interpretable. Read counts from data rather than copying historical cohort
sizes into plotting logic.

With `requirements-notebook.txt` installed, execute:

```bash
bash scripts/reproduce_results.sh
```

This updates saved notebook outputs and needs no model service or LaTeX.
Inspect changed plots for readable labels, legends, clipping and omitted
conditions. Compare exported numbers to the selected source counts, and
check all changed cells for execution errors. Link the executed notebook
and requested figures. Publishing the changed files is a separate action
unless the user also requested synchronization.
