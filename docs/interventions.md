# RQ2 support removal

RQ2 graph edits live in `dataset_split/interventions`; output normalization
and transition metrics live in `eva`:

| Module | Operation |
|---|---|
| `dataset_split.interventions.cwq_witness_breaking` | Condition loss and matched sham removal |
| `dataset_split.interventions.cwq_binding_conflict` | Binding mismatch and matched sham removal |
| `dataset_split.interventions.cwq_progressive_severity` | Fixed-cohort progressive removal |
| `dataset_split.interventions.cwq_rq2_inputs` | Check Grasp graph edits and fixed input population |
| `eva.cwq_rq2_normalize_outputs` | Normalize Final outputs for primary transitions |
| `eva.cwq_progressive_normalize` | Normalize outputs at progressive removal levels |

All take explicit file arguments and expose `--help`:

```bash
python -m dataset_split.interventions.cwq_witness_breaking --help
python -m dataset_split.interventions.cwq_binding_conflict --help
python -m dataset_split.interventions.cwq_progressive_severity --help
python -m eva.cwq_rq2_normalize_outputs --help
python -m eva.cwq_rq2_outcomes --help
python -m eva.cwq_progressive_outcomes --help
```

Construct and freeze the intervention cohort first, run Grasp on each edited
graph with `scripts/run_grasp.sh`, then normalize and score the fixed cohort.
These tools consume only Grasp outputs.

The builders' `--common-pool` input is the registered paper cohort, selected
from questions with correct predictions and sufficient returned evidence for
DoG, GCR*, and Codex. Keep its recorded IDs when reproducing the paper;
replacing it with a newly selected
population changes the experiment. Each record needs `id`,
`common_certified_hit: true`, and `systems.ours.top1`; other systems' predictions
are unnecessary. The builders store that Grasp answer as `system_top1.ours` in
the intervention records. Progressive evaluation reads it directly.

The paper cohort contains 295 condition-loss pairs and 120 binding-mismatch
pairs. Primary evaluation uses these fixed denominators; the two families
pool to 415 intervention pairs. Progressive removal uses 268 questions at
each level. Terminal summaries report APR and ATR with the paper labels.

For a condition-loss bundle:

```bash
python -m dataset_split.interventions.cwq_rq2_inputs \
  --bundle runs/condition_loss --output-dir runs/condition_loss_preflight
python -m eva.cwq_rq2_normalize_outputs \
  --interventions runs/condition_loss/interventions.jsonl \
  --gplus runs/gplus/final.jsonl --critical runs/gminus/final.jsonl \
  --sham runs/gsham/final.jsonl --output runs/condition_loss_predictions.jsonl
python -m eva.cwq_rq2_outcomes \
  --family missing_condition --interventions runs/condition_loss/interventions.jsonl \
  --predictions runs/condition_loss_predictions.jsonl \
  --preflight runs/condition_loss_preflight/preflight.jsonl \
  --output-dir runs/condition_loss_metrics
```

Condition loss removes a minimal supported query pattern while retaining gold
answer nodes. Binding mismatch retains branch-wise answers while eliminating
their shared binding. Sham edits match the deletion count while preserving
support. Progressive removal reuses the same question cohort at every level.
Selection order, graph preprocessing, and fixed denominators retain the existing
research implementation's behavior.
