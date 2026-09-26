#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:?Usage: run_grasp.sh DATASET_JSONL RUN_DIR [retrieval options]}
RUN_DIR=${2:?Usage: run_grasp.sh DATASET_JSONL RUN_DIR [retrieval options]}
shift 2
PYTHON=${PYTHON:-python}
: "${ONTOLOGY_INDEX:?Set ONTOLOGY_INDEX to the relation signature index}"
: "${QWEN_MODEL:?Set QWEN_MODEL to the relation scorer}"
: "${QWEN_REVISION:?Set QWEN_REVISION to the scorer revision}"
: "${GPT_OSS_REVISION:?Set GPT_OSS_REVISION to the serving model revision}"
: "${OLLAMA_BASE_URL:?Set OLLAMA_BASE_URL to the endpoint ending in /v1}"

"$PYTHON" -m src.planning.planner \
  --input-file "$DATASET" --output-dir "$RUN_DIR/plans" \
  --model-revision "$GPT_OSS_REVISION"

"$PYTHON" -m src.main \
  --dataset-path "$DATASET" --plan-dir "$RUN_DIR/plans" \
  --ontology-index "$ONTOLOGY_INDEX" --output-path "$RUN_DIR/retrieval.jsonl" \
  --qwen-model-name "$QWEN_MODEL" --qwen-model-revision "$QWEN_REVISION" \
  --scoring-batch-size "${SCORING_BATCH_SIZE:-8}" "$@"

"$PYTHON" -m src.answering.packing \
  --input-path "$RUN_DIR/retrieval.jsonl" --output-path "$RUN_DIR/packed.jsonl"

final_options=()
if [[ ${FLAT_EVIDENCE:-0} == 1 ]]; then
  final_options+=(--flat-evidence)
fi
"$PYTHON" -m src.answering.reasoner \
  --input-path "$RUN_DIR/packed.jsonl" --output-path "$RUN_DIR/final.jsonl" \
  "${final_options[@]}"
