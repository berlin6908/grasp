"""Evidence-bounded answering over the packed input graph."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from ..contracts import FINAL_JUDGE_REVISION, FINAL_RESULT_VERSION
from ..io import read_jsonl, sha256_file, source_snapshot_sha256, write_json
from ..llm_client import DEFAULT_STRUCTURED_MODEL, call_json
from .prompt import render_flat_judge_prompt, render_judge_prompt, response_schema


def assemble_answer(payload: dict) -> dict:
    answers = list(dict.fromkeys(payload["answers"]))
    if bool(payload["sufficient"]) != bool(answers):
        raise ValueError("Final sufficient must match whether answers are present")
    return {
        "status": "answered" if payload["sufficient"] else "insufficient",
        "answers": answers,
        "reason": payload["reason"],
    }


class FinalReasoner:
    def __init__(
        self,
        model_call: Callable[..., dict] = call_json,
        *,
        flat_evidence: bool = False,
    ):
        self.model_call = model_call
        self.flat_evidence = flat_evidence

    def run(
        self,
        question: str,
        decision: str,
        step_evidence: dict[str, dict],
    ) -> dict:
        prompt = (
            render_flat_judge_prompt(question.strip(), step_evidence)
            if self.flat_evidence
            else render_judge_prompt(
                question.strip(), decision.strip(), step_evidence,
            )
        )
        payload = self.model_call(
            messages=[{"role": "user", "content": prompt}],
            json_schema=response_schema(),
        )
        return assemble_answer(payload)


def run_final_batch(
    reasoner: FinalReasoner, input_path: Path, output_path: Path,
) -> dict:
    """Run final selection once for each successful Pipeline record."""

    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = {"answered": 0, "insufficient": 0, "error": 0}
    error_counts = Counter()
    judge_seconds_total = 0.0
    total = 0
    with output_path.open(
        "x", encoding="utf-8", buffering=1,
    ) as target:
        for upstream in read_jsonl(input_path):
            index = upstream["index"]
            example_id = upstream["example_id"]
            question = upstream["question"]
            decision = upstream["decision"]
            step_evidence = upstream["step_evidence"]

            status = upstream["status"]
            error_type = upstream["error_type"]
            answers = []
            reason = None
            judge_seconds = 0.0
            if status == "ok":
                judge_started = perf_counter()
                try:
                    selection = reasoner.run(
                        question, decision, step_evidence,
                    )
                except Exception:
                    status = "error"
                    error_type = "final_error"
                else:
                    status = selection["status"]
                    answers = selection["answers"]
                    reason = selection["reason"]
                    error_type = (
                        None if status == "answered" else "final_insufficient"
                    )
                judge_seconds = perf_counter() - judge_started

            record = {
                "final_result_version": FINAL_RESULT_VERSION,
                "index": index,
                "example_id": example_id.strip(),
                "question": question.strip(),
                "status": status,
                "error_type": error_type,
                "answers": answers,
                "reason": reason,
                "decision": decision,
                "step_evidence": step_evidence,
                "judge_seconds": judge_seconds,
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            judge_seconds_total += judge_seconds
            status_counts[status] += 1
            if status == "error":
                error_counts[error_type] += 1
            total += 1
    return {
        "total": total,
        "status": status_counts,
        "errors": dict(sorted(error_counts.items())),
        "judge_seconds": judge_seconds_total,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform evidence-bounded answering from the support specification and packed evidence.",
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--flat-evidence", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite final output: {args.output_path}")
    manifest_path = args.output_path.with_suffix(args.output_path.suffix + ".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite final manifest: {manifest_path}")
    started_at = datetime.now(timezone.utc).isoformat()
    input_sha256 = sha256_file(args.input_path)
    source_sha256 = source_snapshot_sha256()
    summary = run_final_batch(
        FinalReasoner(flat_evidence=args.flat_evidence),
        args.input_path,
        args.output_path,
    )
    manifest = {
        "status": "completed_with_errors" if summary["errors"] else "completed",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input_path.resolve()),
        "input_sha256": input_sha256,
        "output_path": str(args.output_path.resolve()),
        "output_sha256": sha256_file(args.output_path),
        "source_snapshot_sha256": source_sha256,
        "configuration": {
            "final_result_version": FINAL_RESULT_VERSION,
            "input": (
                "question_and_flat_packed_triples"
                if args.flat_evidence
                else "question_decision_hint_and_plan_step_evidence"
            ),
            "model": DEFAULT_STRUCTURED_MODEL,
            "model_revision": FINAL_JUDGE_REVISION,
            "output": "sufficient_answers_and_triple_grounded_reason",
        },
        "summary": summary,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
