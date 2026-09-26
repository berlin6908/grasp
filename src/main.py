"""Batch CLI for Grasp specification-guided grounding."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .contracts import (
    ASSIGNMENT_SCORING, DAG_SEARCH, FULL_DAG_TREATMENT,
    NO_BINDING_JOINS_DAG_SEARCH, NO_BINDING_JOINS_TREATMENT,
    NO_SIGNATURE_RELATION_SEARCH, NO_SIGNATURE_TREATMENT,
    RELATION_SEARCH, RESULT_VERSION, TERMINAL_OUTPUT_POLICY,
)
from .io import iter_dataset_samples, sha256_file, sha256_tree, source_snapshot_sha256, write_json
from .grounding.scorer import QwenRelationRanker
from .grounding.search import STATE_BEAM_SIZE, DagCraRetriever
from .grounding.signatures import RelationSignatureIndex
from .pipeline import Pipeline
from .planning.planner import PrecomputedPlanner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Grasp specification-guided grounding over question-local input graphs."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--ontology-index", type=Path, required=True)
    parser.add_argument("--qwen-model-name", required=True)
    parser.add_argument("--qwen-model-revision", required=True)
    parser.add_argument("--scoring-batch-size", type=int, required=True)
    parser.add_argument("--start-example", type=int)
    parser.add_argument("--end-example", type=int)
    treatment = parser.add_mutually_exclusive_group()
    treatment.add_argument("--without-relation-signatures", action="store_true")
    treatment.add_argument("--without-binding-joins", action="store_true")
    return parser.parse_args()


def _run_batch(
    pipeline: Pipeline,
    dataset_path: Path,
    output_path: Path,
    *,
    start_example: int | None = None,
    end_example: int | None = None,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    status_counts = {"ok": 0, "insufficient": 0, "error": 0}
    error_counts: dict[str, int] = {}
    with output_path.open("x", encoding="utf-8", buffering=1) as handle:
        for index, sample in iter_dataset_samples(
            dataset_path,
            start_example=start_example,
            end_example=end_example,
        ):
            example_id, question = sample["id"].strip(), sample["question"].strip()
            graph = sample["graph"]
            q_entities = [value.strip() for value in sample["q_entity"]]
            result = pipeline.run(
                question=question,
                graph=graph,
                q_entities=q_entities,
                example_id=example_id,
            )
            record = {
                "index": index,
                "example_id": example_id,
                "question": question,
                **result,
            }
            status = record["status"]
            error_type = record["error_type"]
            status_counts[status] += 1
            if status == "error":
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
    return {
        "dataset_path": str(dataset_path.resolve()),
        "output_path": str(output_path.resolve()),
        "start_example": start_example,
        "end_example": end_example,
        "total": total,
        "status": status_counts,
        "errors": error_counts,
    }


def _plan_generation_configs(plan_dir: Path) -> list[dict]:
    configs = set()
    for path in sorted(plan_dir.glob("*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        config = artifact["generation"]["config"]
        configs.add(json.dumps(config, sort_keys=True, separators=(",", ":")))
    return [json.loads(config) for config in sorted(configs)]


def main() -> None:
    args = _parse_args()
    if args.scoring_batch_size < 1:
        raise ValueError("--scoring-batch-size must be positive")
    if args.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite pipeline output: {args.output_path}")
    manifest_path = args.output_path.with_suffix(args.output_path.suffix + ".manifest.json")
    base_runtime = {
        "device": "cuda",
        "dtype": "bfloat16",
        "qwen_model": args.qwen_model_name,
        "qwen_model_configured_revision": args.qwen_model_revision,
        "scoring_batch_size": args.scoring_batch_size,
    }
    if args.without_relation_signatures:
        retrieval_treatment = NO_SIGNATURE_TREATMENT
        dag_search = DAG_SEARCH
        relation_search = NO_SIGNATURE_RELATION_SEARCH
    elif args.without_binding_joins:
        retrieval_treatment = NO_BINDING_JOINS_TREATMENT
        dag_search = NO_BINDING_JOINS_DAG_SEARCH
        relation_search = RELATION_SEARCH
    else:
        retrieval_treatment = FULL_DAG_TREATMENT
        dag_search = DAG_SEARCH
        relation_search = RELATION_SEARCH
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.dataset_path.resolve()),
        "dataset_sha256": sha256_file(args.dataset_path),
        "plan_dir": str(args.plan_dir.resolve()),
        "plan_artifacts_sha256": sha256_tree(args.plan_dir, list(args.plan_dir.glob("*.json"))),
        "ontology_index_path": str(args.ontology_index.resolve()),
        "ontology_index_sha256": sha256_file(args.ontology_index),
        "source_snapshot_sha256": source_snapshot_sha256(),
        "configuration": {
            "result_version": RESULT_VERSION,
            "assignment_scoring": ASSIGNMENT_SCORING,
            "dag_search": dag_search,
            "state_beam_size": STATE_BEAM_SIZE,
            "terminal_output_policy": TERMINAL_OUTPUT_POLICY,
            "pipeline_output": "factorized_grounded_program_frames",
            "plan_generation_configs": _plan_generation_configs(args.plan_dir),
            "relation_discovery_graph": "full_question_graph",
            "relation_execution_graph": "full_question_graph",
            "retrieve_unit": "single_directed_graph_relation",
            "retrieval_treatment": retrieval_treatment,
            "relation_signature_mode": (
                "without" if args.without_relation_signatures else "with"
            ),
            "relation_search": relation_search,
        },
        "runtime": base_runtime,
    }
    write_json(manifest_path, manifest)
    try:
        ontology = RelationSignatureIndex.load(args.ontology_index)
        ranker = QwenRelationRanker(
            args.qwen_model_name,
            args.qwen_model_revision,
            args.scoring_batch_size,
        )
        manifest["runtime"].update(ranker.runtime_metadata())
        pipeline = Pipeline(
            planner=PrecomputedPlanner(args.plan_dir),
            retriever=DagCraRetriever(
                ranker,
                use_signatures=not args.without_relation_signatures,
                enforce_intersections=not args.without_binding_joins,
            ),
            ontology=ontology,
        )
        summary = _run_batch(
            pipeline,
            args.dataset_path,
            args.output_path,
            start_example=args.start_example,
            end_example=args.end_example,
        )
    except BaseException as error:
        manifest.update(
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=f"{type(error).__name__}: {error}",
        )
        write_json(manifest_path, manifest)
        raise
    manifest.update(
        status=("completed_with_errors" if summary["errors"] else "completed"),
        finished_at=datetime.now(timezone.utc).isoformat(),
        output_path=str(args.output_path.resolve()),
        output_sha256=sha256_file(args.output_path),
        summary=summary,
    )
    write_json(manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
