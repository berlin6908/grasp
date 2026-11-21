---
name: grasp-development
description: Implement, debug or review changes to the Grasp research pipeline while preserving its experimental semantics and simple module boundaries. Use for code changes in src, dataset_split, eva or their command-line entrypoints.
---

# Grasp development

Start with the requested behavior and the code it reaches. Search callers
with `rg`, inspect the corresponding shell entrypoint, and read the installed
dependencies in [pyproject.toml](../../pyproject.toml) before adding a package
or replacing a library feature.

## Reuse the current boundaries

| Area | Responsibility |
|---|---|
| `src/planning/` | Support specification model, prompt, generation and loading |
| `src/grounding/` | Graph, bindings, signatures, relation scoring and joint search |
| `src/answering/` | Evidence packing, answering prompt and Final generation |
| `src/pipeline.py`, `src/main.py` | Stage orchestration and grounding CLI |
| `src/io.py`, `src/llm_client.py` | Shared serialization and structured model access |
| `dataset_split/` | Benchmark annotations and graph-relative support classification |
| `dataset_split/interventions/` | Registered condition-loss, binding and progressive edits |
| `eva/` | Answer/evidence scoring, replay and fixed-population aggregation |

The published package exposes `src`, `dataset_split` and `eva` directly.
Update current imports, callers, docs and shell arguments when changing an
interface. Remove superseded code instead of introducing legacy aliases,
migrations or fallback paths. Keep a working end-to-end route while changing
one concern at a time. Prefer existing helpers and maintained dependencies
to new wrappers or configuration layers.

## Preserve the research contract

Predicted support specifications and evaluation-only constraints have
different roles. Do not let evaluation labels, gold answers or partition
membership enter prediction decisions. A retrieved edge is not automatically
Final-cited evidence. Keep shared-binding semantics, stable selected
populations and the distinction between Error and Unknown.

Serialized names such as `Plan`, `Final`, `step_evidence` and internal K/U
labels remain meaningful interfaces; paper prose uses the terminology in
the current README and evaluation docs. User-requested semantic changes
should be recorded as new experimental conditions rather than silently
relabeling frozen outputs.

## Check the behavior that changed

For a bug fix, reproduce the failure and fix the common cause after reading
its callers. A nontrivial logic change needs a small behavioral regression
check or assert-based self-check. Test observable outcomes, binding/evidence
semantics or a real failure; do not add checks that merely require a word,
field or helper to exist. Documentation-only changes need no runtime tests.

Use the existing synthetic workflow `bash scripts/smoke.sh runs/dev_smoke`
when split/evaluation behavior is affected. The publication does not ship
the private development test suite, and the synthetic workflow does not
exercise GPU inference. For model-path changes, test the affected operation
with a suitable local fixture or authorized representative model run.

Review the final diff for correctness, simplicity, dependency direction and
preserved error handling. State what changed, what was checked, and the
limits of that check. Avoid unrelated refactors and extra infrastructure.
