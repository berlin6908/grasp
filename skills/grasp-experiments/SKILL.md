---
name: grasp-experiments
description: Run or continue Grasp planning, joint grounding, evidence packing, answering and evaluation with the repository entrypoints. Use for an explicitly requested experiment or component comparison on supplied data and model infrastructure.
---

# Grasp experiments

Use the [root README](../../README.md) and [data instructions](../../docs/data.md)
to resolve the dataset, model revisions, endpoint, signature index and output
directory. This release contains Grasp execution, not baseline runners or a
cluster deployment. Use the user's supplied infrastructure and the named run's
configuration; no host, account or scheduler is assumed by this skill.

## Existing pipeline

Run commands from the repository root. `scripts/run_grasp.sh` connects:

1. `src.planning.planner` → `plans/`.
2. `src.main` → `retrieval.jsonl`.
3. `src.answering.packing` → `packed.jsonl`.
4. `src.answering.reasoner` → `final.jsonl`.

The runner requires `OLLAMA_BASE_URL` ending in `/v1`, `GPT_OSS_REVISION`,
`QWEN_MODEL`, `QWEN_REVISION` and `ONTOLOGY_INDEX`. Read their recorded values
from the selected configuration and current README. Grounding requires CUDA
with bfloat16, and the scorer and packing tokenizer use the local model cache.
Keep credentials out of saved configurations and version control.

After the actual data and environment are selected:

```bash
bash scripts/run_grasp.sh "$DATASET_JSONL" "$RUN_DIR"
```

Use a separate output directory for a new condition. For a continuation,
inspect the completed stage artifacts and the individual stage's output
behavior before invoking it; the all-stage shell runner is not a generic
resume controller. A saved retrieval or packed file is not a completed
answering or evaluation stage.

## Controlled conditions

Reuse the existing options rather than adding launch wrappers:

| Requested condition | Runner setting |
|---|---|
| Without relation guidance | `--without-relation-signatures` |
| Without binding consistency | `--without-binding-joins` |
| Without specification-aware answering | `FLAT_EVIDENCE=1` |

The two grounding switches are mutually exclusive in the current CLI.
For 2WikiMultiHopQA's recorded relation-only configuration, use
`examples/relation_signatures_none.json` as `ONTOLOGY_INDEX` and pass
`--without-relation-signatures`.

When comparing components or scorer sizes, retain the reference population,
plans, non-treatment settings, answer model, schema delivery and parser.
A change in model service or output protocol is a new condition, even if
both outputs parse as JSON. Identify the scorer separately from the planning
and answering model.

## Finish the requested stages

For a new full run, exercise the actual entrypoint on a representative small
input before the full population. The CPU-only `scripts/smoke.sh` checks split
and evaluation behavior; it does not test GPU inference or the model service.
Reuse valid existing receipts when the frozen condition is unchanged.

Connect saved Final output to `scripts/evaluate.sh` with the corresponding
annotations and partition index. Report stage completion, evaluated counts
and comparison scope separately. A result request ends at completed canonical
evaluation; an explicitly stage-limited request ends at that stage. A status
or figure request alone does not authorize launching an experiment.
