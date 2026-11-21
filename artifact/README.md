# Paper artifacts

The original paper-selected bundle is dated 2026-09-26. The
[supplementary experiments](supplement/README.md), added 2026-11-21, include
**GNN-RAG, SubgraphRAG and RJE**, additional **CoT/GCR*/DoG** conditions,
reused **Grasp** references and the specification-quality audit. Each recorded
condition keeps its own model, protocol and population.

[index.json](index.json) lists the conditions, paper references, and available
records. [sources.tsv](sources.tsv) records anonymous source identifiers and the
selections used to produce this bundle. The same identifier denotes the same
recorded source; private workspace paths and host identifiers are omitted.

Run [local-results.ipynb](../local-results.ipynb) to inspect and redraw the selected
paper tables and statistical figures using only these bundled files.

```text
main/           Three datasets, five systems, and the reported model settings
ablation/       Grasp components and grounding model sizes
interventions/  Condition loss, binding mismatch, and progressive removal
analysis/       Semantic relation relaxation and partition relaxation
splits/         Paper S/I membership and CWQ curation labels
paper/          Current table cells and plot data
supplement/     Completed additional comparisons, plan audit, and diagnostics
```

The original bundle contains **80 conditions**: 39 natural-graph conditions, 9 component
ablations, 2 additional grounding model sizes, 26 primary interventions, and
4 progressive-removal conditions. The 7B size reference reuses the main Grasp
result. Codex results are stored once per dataset.

The index now has **109 run entries**: those 80 plus 29 supplementary natural-input
conditions. Eight supplementary entries also include matched support-removal
and progressive records. The audit and diagnostic analyses are indexed separately.
Supplementary record files use the paper S/I populations; their canonical source
summaries can additionally describe the full registered evaluations.

The analysis files contain the paper's 80 semantic-relaxation curve points
on the fixed CWQ S split and eight partition-relaxation settings. Structure
data contain the three plotted groups: chain, branching, and intersection.

## Populations

| Dataset | S | I | Total |
|---|---:|---:|---:|
| CWQ | 1,679 | 1,459 | 3,138 |
| WebQSP | 1,335 | 286 | 1,621 |
| 2WikiMultiHopQA | 1,383 | 2,614 | 3,997 |

CWQ uses final `mismatch=no` IDs. Its complete curation labels document the
appendix's original/reported population comparison. Primary interventions
contain 295 condition-loss pairs and 120 binding-mismatch pairs; progressive
removal uses 268 questions at each level. The two primary families may share
question IDs, so their pooled denominator is 415 intervention pairs.

## Reading the files

- `details.jsonl`, `outcomes.jsonl`, and available `final.jsonl` preserve the
  selected per-question records. Stored K/U mean paper S/I. DoG's
  `supported_hit` is identified in the index as its evidence-hit field.
- `metrics.json` contains counts and fractional rates, or explicitly marked
  paper percentages. CSV tables under `paper/` use percentages.
- `configuration.json` and `protocol.json` contain the settings of the
  selected experiments. Grasp model labels identify the grounding scorer;
  support specification induction and evidence-bounded answering use GPT-OSS-120B.
- Large per-question files use standard `.jsonl.gz` compression for
  anonymous hosting. `index.json` lists them under `storage`.
  Decompress the selected input before using the JSONL command-line tools:

  ```bash
  gzip -dk artifact/main/cwq/grasp/qwen/final.jsonl.gz
  ```

  The results notebook reads the uncompressed tables and summaries directly.
- Four Grasp conditions—Gemma and Llama on WebQSP and 2WikiMultiHopQA—have recorded
  summaries only. The two additional scorer sizes have complete evaluation
  rows but no local Final payloads. The index identifies available files.
- DoG 2WikiMultiHopQA includes native predictions and exact evaluator counts.
  Input graphs are supplied through the [dataset instructions](../docs/data.md).

## Reporting conventions

CWQ radar data use the current main-table conditions, including corrected
Gemma-CoT. WebQSP Grasp/Qwen follows the paper's selected support and abstention
sources; its matching support rows and published aggregate counts are included.

Intervention results retain the reviewed classifications used by the paper;
their decision sources are listed in `sources.tsv`. Per-question errors remain
distinct. The current DoG transition table includes these errors in its
displayed Other total; `paper/support_removal_table.csv` preserves that display.

Counts and memberships were checked against the selected tables and plot data.
No new model inference or prediction rescoring was performed.
