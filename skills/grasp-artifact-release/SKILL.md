---
name: grasp-artifact-release
description: Package completed Grasp or comparison-system experiment results into artifact, update the source index and executed notebook, and synchronize an explicitly requested repository release. Use for adding supplementary results or publishing result bundles.
---

# Grasp artifact release

Read [artifact/index.json](../../artifact/index.json),
[artifact/README.md](../../artifact/README.md) and the selected experiment's
completion/evaluation records before copying files. This repository ships
Grasp code plus comparison-system results. Baseline implementations, model
weights and private experiment infrastructure are separate.

## Select completed, traceable results

Resolve the system, answer model, retriever/scorer, graph source, prompt arm,
population and canonical evaluation. Retrieval completion alone is not an
evaluated result. Reuse frozen summaries and predictions; packaging does not
require model execution or rescoring. Record incomplete conditions as
unavailable without turning partial output into a full score.

Use the existing layout for the result family. `artifact/paper/` retains the
selected original paper view; supplementary conditions live under
`artifact/supplement/`. Keep genuinely different recorded conditions distinct,
and replace a superseded export only when it represents the same intended
condition. Do not duplicate a reference solely to inflate a condition count.

For a supplementary natural condition, include:

- `metrics.json`: selected S/I counts, denominators, rates and condition identity.
- `details.jsonl.gz`: the matching selected per-question evaluation records.
- `source_summary.json`: the canonical source summary, whose full population
  may differ from the selected S/I record set.

Describe summary-only or missing-Final availability accurately. Keep Error
and Unknown distinct. Matched primary transitions use the same arm's G+;
progressive transitions use its k=0, not separate natural-input predictions.
Preserve model-assisted audit verdicts and confidence without inventing a
cutoff or human-annotation claim.

## Update the publication together

Extend `artifact/index.json` and `artifact/sources.tsv` alongside the data.
Keep real workspace/host paths in any private source map outside the checkout;
the publication uses anonymous source identifiers. Retain scientific model,
protocol and third-party source information. Omit credentials, account/host
identifiers, private logs and unrelated files.

Use standard gzip for large per-question files, as the current bundle does.
Update the index's compressed-file list and validate that new data can be
read. Check the actual anonymous host's current file constraints only if that
hosting service is part of the requested publication.

Update the relevant README and notebook sections so the added results are
discoverable and reproducible. Retain CSV units and JSON rate units. Before
publication, compare selected IDs and integer metric counts against the
included per-question records; check intervention pairing and denominators.
For audit additions, verify verdict-group counts and the downstream strata.
Reuse unchanged frozen validation receipts instead of rescanning unrelated
experiments. Execute changed notebook cells through the notebook workflow
and visually inspect new or changed figures.

## Synchronize the requested destination

Review and stage only the intended release files. When the user requested
synchronization, commit and push the named remote/branch and verify its actual
commit and added artifact index. Preserve unrelated local work and existing
history. Rebuild an existing local release archive when it is part of the
maintained deliverable.

A fixed anonymous snapshot is independent of the Git branch. If anonymous
publication is requested, update the selected snapshot through its authorized
management session and verify the displayed result. A successful Git push
does not establish that the anonymous mirror changed. Report a missing login
or other concrete blocker separately from completed GitHub synchronization.
