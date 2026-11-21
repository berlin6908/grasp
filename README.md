# Grasp

Grasp (**Grounding with Requirement-Aware Shared-binding Preservation**) is a
training-free framework with three stages: **support specification induction**,
**specification-guided grounding**, and **evidence-bounded answering**. It operates
on the input knowledge supplied as a question-local graph. This distribution
contains the S/I split builders, Grasp implementation, support-removal interventions, and
paper evaluation. Only Grasp code is included. The accompanying `artifact/`
directory also contains the comparison-system results used in the paper.

The [supplementary experiments](artifact/supplement/README.md), updated
2026-11-21, add 29 natural-input conditions, eight matched prompt arms with
support-removal results, and an 8,756-question specification-quality audit.
The results notebook includes these comparisons and their recorded protocols.

Start with the executed English
[results notebook](local-results.ipynb). Its CPU-only setup is described below
under **Reproduce the paper tables and figures**.

## Layout

```text
dataset_split/      CWQ, WebQSP, and 2WikiMultiHopQA S/I/X construction
  interventions/    RQ2 support removal and edited graph construction
src/
  planning/
    plan.py         Support specification (Plan) data model and JSON encoding
    prompt.py       Support specification induction prompt and response schema
    planner.py      Support specification generation, loading, and CLI
  grounding/
    graph.py        Input-graph index and directed relation traversal
    state.py        Bindings, evidence, assignments, and join operations
    signatures.py   Freebase relation signatures and index-building CLI
    prompt.py       Relation-scoring prompt
    scorer.py       Frozen relation scorer
    search.py       Specification-guided grounding with beam search
  answering/
    packing.py      Evidence selection, token budget, and packing CLI
    prompt.py       Answer prompt, citations, and response schema
    reasoner.py     Answer generation and Final CLI
  pipeline.py       Connect saved support specifications to grounding and evidence
  main.py           Grounding run CLI
  io.py             Shared artifact reading, writing, and provenance
  llm_client.py     Shared structured model client
  contracts.py      Recorded protocol constants
eva/                Answer scoring, evidence replay, paper metrics,
                    and RQ2 result normalization
artifact/           Paper experiment results, splits, plot data, and sources
scripts/            Bash entry points
local-results.ipynb  Executed notebook for paper tables and statistical figures
requirements-notebook.txt  CPU-only notebook dependencies
pyproject.toml      Package dependencies
```

The three main directories follow the paper workflow: construct the input
graphs and splits, run Grasp, then evaluate its predictions. The system imports
as `src`; split and evaluation tools import as `dataset_split`
and `eva`. Shell scripts launch the stages. The split and evaluation packages
reuse the shared JSONL helpers
in `src.io`.

As in the paper's prompt appendix, `Plan` is the implementation name for the
complete predicted support specification P. `Evidence DAG` in the grounding
prompt refers to its retrieval and intersection steps, excluding the final
answering instruction. These names and the serialized `Plan`/`Final` fields
are retained. Evaluation-only constraints come from benchmark annotations
and are distinct from the predicted support specification.

The release uses the current artifact schemas directly. Internal stages assume
their inputs were produced by the preceding stage; semantic checks protect
valid support specifications, evidence replay, and fixed evaluation populations.

## Install

Use Python 3.10 or newer in a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

All three packages install together through setuptools package discovery.
Use Bash on Linux/macOS or Git Bash/WSL on Windows. Invoke scripts from this
directory. `PYTHON` can select the interpreter; it defaults to `python`.

Run the local data-split/evaluation example without model downloads or API calls:

```bash
bash scripts/smoke.sh runs/toy
```

The example uses synthetic graphs and saved illustrative answers. It checks
split construction and evaluation; it does not measure model quality.

## Prepare data

Use the source datasets, graph collections, official annotations, and entity
alignment resources described in [docs/data.md](docs/data.md). Supply paths
explicitly. Dataset payloads and model weights are separate downloads.

```bash
bash scripts/split.sh cwq --help
bash scripts/split.sh webqsp --help
bash scripts/split.sh 2wiki --help
```

CWQ example:

```bash
bash scripts/split.sh cwq \
  --rog-source data/cwq.jsonl --rog-split test --rog-revision main \
  --original-cwq data/ComplexWebQuestions_test.json \
  --nsm-test data/nsm/test.json --nsm-entities data/nsm/entities.txt \
  --reverse-properties data/reverse_properties --output-dir runs/cwq_split
```

2WikiMultiHopQA example:

```bash
bash scripts/split.sh 2wiki \
  --dataset data/2wiki_graphs.jsonl --official-dev data/2wiki/dev.json \
  --official-aliases data/2wiki/id_aliases.jsonl --output-dir runs/2wiki_split
```

For CWQ/WebQSP, build a Freebase relation-signature index for the selected
input graph collection:

```bash
python -m src.grounding.signatures \
  --fb-roles data/fb_roles --dataset-path data/cwq.jsonl \
  --output-path runs/relation_signature_index.json
```

## Run Grasp

Specification-guided grounding requires a CUDA GPU supporting bfloat16. The
scorer and packing tokenizer use the local Hugging Face cache. Cache the
specified revisions first.
Support specification induction and evidence-bounded answering use the
OpenAI-compatible `gpt-oss:120b` service.

```bash
export OLLAMA_BASE_URL=http://HOST:PORT/v1
export GPT_OSS_REVISION="$(python -c 'from src.contracts import FINAL_JUDGE_REVISION; print(FINAL_JUDGE_REVISION)')"
export QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
export QWEN_REVISION=a09a35458c702b33eeacc393d103063234e8bc28
export ONTOLOGY_INDEX=runs/relation_signature_index.json
export SCORING_BATCH_SIZE=8
bash scripts/run_grasp.sh data/cwq.jsonl runs/grasp_cwq
```

The script produces `plans/`, `retrieval.jsonl`, `packed.jsonl`, and `final.jsonl`.
Use a new output directory for each condition. Packing uses
`openai/gpt-oss-120b` tokenizer revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a` with a 4,096-token input budget.

2WikiMultiHopQA uses text relations and the recorded relation-only configuration.
After setting the model/service variables above, select the empty signature
index and disable Freebase signature guidance:

```bash
export ONTOLOGY_INDEX=examples/relation_signatures_none.json
bash scripts/run_grasp.sh data/2wiki_graphs.jsonl runs/grasp_2wiki \
  --without-relation-signatures
```

The paper components use the same code paths:

- **w/o relation guidance:** append `--without-relation-signatures`.
- **w/o binding consistency:** append `--without-binding-joins`.
- **w/o specification-aware answering:** set `FLAT_EVIDENCE=1`.

Stages can also run individually via `python -m`; each exposes `--help`.

## Evaluate

Evaluation takes Grasp Final records and an explicit partition index. It does
not require baseline predictions. Rates are fractions; multiply by 100 for
percentage display. See [docs/evaluation.md](docs/evaluation.md) for definitions.
Pass `--flat-evidence` when evaluating the component without specification-aware answering.

```bash
bash scripts/evaluate.sh --benchmark cwq \
  --dataset data/cwq.jsonl --final runs/grasp_cwq/final.jsonl \
  --partition-index runs/cwq_split/witness_details.jsonl \
  --original-cwq data/ComplexWebQuestions_test.json \
  --witness-details runs/cwq_split/witness_details.jsonl \
  --reverse-properties data/reverse_properties --output-dir runs/cwq_metrics
```

For WebQSP, pass `--official-webqsp`, `--supplemental-mid-names`, and
`--reverse-properties`. For 2WikiMultiHopQA, pass the split builder's
`2wiki_required_evidence_closed_world_v08_labels.jsonl` as `--partition-index`.
`summary.json` reports the six paper measures; `details.jsonl` preserves
per-question decisions, including errors and unresolved rows.

For a curated paper population, supply the matching curated dataset and partition
index. Every Final ID must belong to that dataset; select the corresponding Final
records before evaluating a subset. Missing predictions remain in the denominator.

RQ2 construction and APR/ATR commands are in [docs/interventions.md](docs/interventions.md).

## Paper artifacts

[artifact/README.md](artifact/README.md) indexes the paper results for Grasp,
CoT, GCR*, DoG, and Codex. It contains 80 experimental conditions, the reported
S/I populations, semantic-relaxation records, and data for the paper figures.
Available Final and evaluation records use the paper's selected IDs. Each
condition keeps one selected result, and plot data use the current main-table
values. The index identifies summary-only conditions and reporting conventions.
Dataset graph collections and model weights remain separate downloads.

## Reproduce the paper tables and figures

[local-results.ipynb](local-results.ipynb) is the English results notebook for
Grasp. It includes saved outputs and reads only the bundled `artifact/`
files. It covers the selected S/I populations, RQ1/RQ2/RQ3 results, component and
model-size analyses, and semantic-relaxation results. See the notebook for metric
definitions and the selected reporting conventions.

In a Python environment, install the notebook dependencies and run:

```bash
python -m pip install -r requirements-notebook.txt
python -m notebook local-results.ipynb
```

Choose **Kernel → Restart & Run All**, or execute from the release directory:

```bash
bash scripts/reproduce_results.sh
```

The script updates the notebook's saved outputs. Tables and curve data are
exported as CSV, and figures as PDF/PNG, under `runs/paper-results/`.
This CPU-only workflow requires no core-package installation, model downloads,
API credentials, or LaTeX. It redraws the selected experimental results;
model execution and prediction evaluation use the separate commands above.
