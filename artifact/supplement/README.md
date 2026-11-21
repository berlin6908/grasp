# Supplementary experiments

Updated 2026-11-21 from completed evaluations. These records extend the
September paper bundle with **29 natural-input conditions**, **8 matched
prompt arms with support-removal results**, and the **8,756-question support
specification audit**. The 29 conditions include two reused Grasp references.
No model inference or prediction rescoring was performed for this release.

Open `local-results.ipynb` in the repository root and run all cells. Sections
7–9 display the supplementary comparisons, audit, and current-condition
diagnostics using only the bundled files. The original paper tables retain
their original selected conditions.

## Contents

| Path | Contents |
|---|---|
| `natural_metrics.csv` | Six natural-input metrics, integer numerators and denominators; display values are percentages |
| `natural/` | GNN-RAG, SubgraphRAG and RJE original/Unknown results; cross-dataset CoT/GCR* controls; official-source DoG 120B; reused Grasp references |
| `prompt_controls/cwq/` | Matched default/Unknown CoT Qwen, CoT 120B, DoG Qwen and GCR* arms, including natural and removal records |
| `support_removal.csv` | 32 condition-loss/binding-mismatch × critical/sham rows across the eight arms |
| `progressive.csv` | 24 rows: eight arms at each of k=0,1,2 |
| `specification_audit/` | Raw three-way judgments, downstream joins, rubric, prompt, schema, quality reports and confidence sensitivity |
| `diagnostics/` | 112 relation-relaxation points for 14 recorded configurations, and constraint-count/structure strata |

Each natural condition has `metrics.json`, `source_summary.json`, and
`details.jsonl.gz`. The first is the selected S/I view, verified against its
per-question records. The source summary preserves the canonical evaluation,
including any original full-population diagnostics. Private path metadata is
replaced by source identifiers. `../index.json` lists each condition and
`../sources.tsv` records its provenance and selection.

## Populations and outcome states

The natural record files contain the same frozen paper IDs as `../splits/`:
CWQ S/I = 1,679/1,459; WebQSP = 1,335/286; 2WikiMultiHopQA = 1,383/2,614.
CWQ excludes the 393 curation exclusions; WebQSP and 2WikiMultiHopQA exclude
7 and 19 unresolved rows. The source summaries can also describe those full
registered populations. Each condition records both source and selected row
counts. Errors remain in the selected denominator and receive no Unknown credit.

Hit@1 and EvidenceHit@1 divide by S. EvidenceGap is the fraction of correct S
predictions without returned support. CAR_I and OAR_S are Unknown rates on I
and S. EBA combines supported-correct S predictions with Unknown I responses,
dividing by S + I. CSV metric values are percentages; JSON `rate` values are
fractions. Integer counts and denominators are retained.

Support removal keeps 295 condition-loss pairs, 120 binding-mismatch pairs
and 268 progressive questions. The pooled denominator is 415 pairs across
295 distinct questions. Transitions use **the same arm's paired G+ answer**
for condition loss/binding mismatch and **its k=0 answer** for progressive
removal. Separate natural-input outputs are not the pairing reference.
`same_answer`, `other_answer`, `abstention`, and `error`
remain separate. APR is the `same_answer` rate. These are canonical observed
outcomes; the September paper's reviewed display assignments are not applied
to the new prompt controls. Source summaries retain sham contrasts and
their recorded confidence intervals.

## Conditions that affect interpretation

- **Matched CoT Qwen:** fresh default/control natural runs use the same
  512-token cap; removal uses 1,024. The historical main-table natural launch
  override is unrecorded. These are a matched pair, separately identified
  from the historical CoT row. The 129 unavailable CWQ graphs remain errors.
- **Matched CoT 120B:** both arms use GPT-OSS-120B, 4,096 total output tokens,
  temperature 0, seed 0 and medium reasoning. The default no-abstention rule
  is replaced with the common evidence-only/Unknown instruction in control.
- **GCR*:** Qwen2.5-7B generates frozen paths; GPT-OSS-120B generates answers.
  The CWQ control changes the answering instruction on the same paths. The
  recorded default is reused with the canonical answer normalization.
  WebQSP and 2WikiMultiHopQA also use the recorded Qwen2.5-7B path model with
  GPT-OSS-120B answering.
- **Matched DoG Qwen:** default and control share native interface corrections
  and frozen source settings. This is separate from historical DoG and from
  the new 120B condition.
- **Official-source DoG 120B:** native constrained decoding at upstream commit
  `de0df080ab82f4dbd7615471f2ad33fed5fbf78a`, with one added Unknown sentence.
  The MXFP4 checkpoint revision is `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`;
  beam size is 1, graph steps are 1–5, and the cap is 4,096 tokens per phase.
  Returned evidence is the full consecutive numbered triple prefix, including
  native free final generation. The earlier adapted native condition and
  cancelled API pilot are not substituted for this selected condition.
- **GNN-RAG and SubgraphRAG:** their native retrievers and evidence interfaces
  are retained. SubgraphRAG uses the authors' cached graphs, which differ from
  the shared graph collection; its inherited S/I grouping is supplementary.
- **RJE:** evidence is the terminal graph context displayed to the reader.
  The Unknown treatment also removes permission to fill knowledge gaps and
  changes intermediate sufficiency instructions. WebQSP uses the CWQ-trained
  retriever/ranker. It is not an isolated one-sentence prompt effect.
- **Cross-dataset 120B controls:** CoT/GNN-RAG use 4,096 output tokens,
  SubgraphRAG 4,000 and RJE 2,048; GCR* retains native client options. These are
  method-specific budgets. Missing CoT graphs (80 WebQSP, 336 2WikiMultiHopQA)
  remain input failures in their registered populations.
- **Reused Grasp:** these are existing 120B answers. The supplementary WebQSP
  snapshot has 163 S and 158 I Unknown responses; the September main table
  uses its separately documented selected reporting sources. The 2WikiMultiHopQA
  source records Final v13, without a retained historical runtime digest.

## Specification quality audit

The completed v4.1 model-assisted audit uses `gpt-6.1-sol` and includes all
8,756 original plans, including 20 invalid plans. It retains correct,
incorrect and uncertain judgments, with no confidence cutoff. The rubric,
judge prompt and decision schema accompany the raw judgments.

Confidence is the judge's strength of belief in a decisive verdict. A high
score can accompany an incorrect-plan verdict; it is not a calibrated
probability that the plan is correct. These are model judgments, not human
annotations. Downstream labels were joined after judging. Each of the nine
dataset/verdict strata uses its own S/I denominators; the results describe
association with unchanged predictions, not the effect of repairing plans.

## Current-condition diagnostics

The relation-relaxation data hold the original 1,679-question CWQ S population,
predictions and returned evidence fixed across exact matching and seven
thresholds. Original errors and Unknown responses stay fixed. The data use
the same recorded evidence interfaces as their natural-input conditions.
Original CoT/DoG use Qwen and their displayed Unknown conditions use 120B;
those curves compare recorded configurations. They are not matched
prompt-only comparisons. The constraint-count and structure data retain the
same condition labels and source counts.

## Results not yet available in this bundle

RJE and GNN-RAG CWQ support-removal runs have no completed canonical evaluation
and local-return receipt in the inspected workspace. CoR Unknown controls
remain paused; 2WikiMultiHopQA GNN-RAG, SubgraphRAG and RJE transfer controls
were cancelled. No score or partial prediction set is published for these
unfinished conditions.
