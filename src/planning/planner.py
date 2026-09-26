"""Induce support specifications and load saved Plan artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from ..io import iter_dataset_samples, source_snapshot_sha256, write_json
from ..llm_client import DEFAULT_STRUCTURED_MODEL, call_json
from .plan import PlanDocument, plan_from_payload, plan_from_planner_payload, plan_to_payload
from .prompt import build_planner_contract, build_planner_messages, planner_response_schema


class PlanNotFoundError(KeyError):
    pass


class PlanInvalidError(ValueError):
    pass


class PlanInfrastructureError(RuntimeError):
    pass


class PrecomputedPlanner:
    def __init__(self, artifact_dir: Path):
        self.plans, self.errors = self._load(artifact_dir)

    def run(self, example_id: str, question: str) -> PlanDocument:
        if example_id in self.errors:
            artifact_question, message = self.errors[example_id]
            self._validate_question(example_id, artifact_question, question)
            raise PlanInvalidError(message)
        try:
            artifact_question, plan = self.plans[example_id]
        except KeyError as error:
            raise PlanNotFoundError(f"No Plan artifact for example: {example_id}") from error
        self._validate_question(example_id, artifact_question, question)
        return plan

    @staticmethod
    def _validate_question(example_id: str, expected: str, actual: str) -> None:
        if expected != actual.strip():
            raise PlanInfrastructureError(
                f"Plan artifact question differs for example {example_id}"
            )

    @staticmethod
    def _load(artifact_dir: Path):
        plans: dict[str, tuple[str, PlanDocument]] = {}
        errors: dict[str, tuple[str, str]] = {}
        paths = sorted(artifact_dir.glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"Plan artifact directory contains no JSON: {artifact_dir}")
        for path in paths:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            example_id = artifact["id"].strip()
            question = artifact["question"].strip()
            if example_id in plans or example_id in errors:
                raise ValueError(f"Duplicate Plan artifact id: {path}")
            if path.name.endswith(".error.json"):
                errors[example_id] = (question, artifact["error"])
                continue
            plans[example_id] = (question, plan_from_payload(artifact["plan"]))
        return plans, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Induce one support specification (Plan) per question."
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-example", type=int)
    parser.add_argument("--end-example", type=int)
    parser.add_argument("--model-revision", required=True)
    return parser.parse_args()


def generate_plan_payload(
    question: str,
    *,
    linked_entities: list[str],
    generation_log: dict[str, Any],
) -> dict[str, Any]:
    contract = build_planner_contract(linked_entities)
    messages = build_planner_messages(question, contract)
    serialized_messages = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    generation_log["prompt_hash"] = hashlib.sha256(
        serialized_messages.encode("utf-8")
    ).hexdigest()
    started = perf_counter()
    try:
        payload = call_json(
            messages,
            json_schema=planner_response_schema(contract),
            usage_callback=generation_log.update,
        )
        plan = plan_from_planner_payload(payload, contract)
    except (KeyError, TypeError, ValueError) as error:
        raise PlanInvalidError(str(error)) from error
    finally:
        generation_log["seconds"] = perf_counter() - started
    return plan_to_payload(plan)


def _generation_metadata(
    *,
    model: str,
    model_revision: str,
    generation_log: dict[str, Any],
    source_sha256: str,
) -> dict[str, Any]:
    return {
        **generation_log,
        "model": model,
        "model_revision": model_revision,
        "source_snapshot_sha256": source_sha256,
        "config": {
            "planner_calls": 1,
            "temperature": 0.0,
            "seed": 0,
            "reasoning_effort": "medium",
            "max_tokens": 4096,
            "sdk_max_retries": 0,
            "strict_json": True,
        },
    }


def main() -> None:
    args = parse_args()
    model = DEFAULT_STRUCTURED_MODEL
    source_sha256 = source_snapshot_sha256()

    success = plan_invalid = 0
    for idx, sample in iter_dataset_samples(
        args.input_file,
        start_example=args.start_example,
        end_example=args.end_example,
    ):
        stem = f"{idx:06d}"
        plan_path = args.output_dir / f"{stem}.json"
        error_path = args.output_dir / f"{stem}.error.json"
        if plan_path.exists() or error_path.exists():
            raise FileExistsError(f"Refusing to overwrite Plan artifact {stem}")

        question = sample["question"].strip()
        qid = sample["id"].strip()
        linked_entities = list(dict.fromkeys(value.strip() for value in sample["q_entity"]))
        generation_log: dict[str, Any] = {}
        artifact = {"id": qid, "question": question}
        output_path = plan_path
        try:
            artifact["plan"] = generate_plan_payload(
                question,
                linked_entities=linked_entities,
                generation_log=generation_log,
            )
        except PlanInvalidError as exc:
            artifact.update(error=f"{type(exc).__name__}: {exc}", error_type="plan_invalid")
            output_path = error_path
            plan_invalid += 1
            message = "plan_invalid"
        else:
            success += 1
            message = "saved"
        artifact["generation"] = _generation_metadata(
            model=model,
            model_revision=args.model_revision,
            generation_log=generation_log,
            source_sha256=source_sha256,
        )
        write_json(output_path, artifact)
        print(f"[{idx}] {message} {stem}")

    print(f"success={success}")
    print(f"plan_invalid={plan_invalid}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
