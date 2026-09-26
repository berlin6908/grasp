#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:-runs/toy}
bash scripts/split.sh 2wiki \
  --dataset examples/toy/graphs.jsonl --official-dev examples/toy/official.json \
  --official-aliases examples/toy/aliases.jsonl --output-dir "$RUN_DIR/split"
bash scripts/evaluate.sh --benchmark 2wiki \
  --dataset examples/toy/graphs.jsonl --final examples/toy/final.jsonl \
  --partition-index "$RUN_DIR/split/2wiki_required_evidence_closed_world_v08_labels.jsonl" \
  --output-dir "$RUN_DIR/evaluation"
