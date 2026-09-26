"""Pack a token-bounded evidence subgraph for the Final answer generator."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from transformers import AutoTokenizer

from ..io import read_jsonl, sha256_file, source_snapshot_sha256, write_json, write_jsonl
from .prompt import render_judge_prompt


PACKING_RESULT_VERSION = 14


JUDGE_INPUT_TOKEN_BUDGET = 4_096


JUDGE_TOKENIZER_MODEL = "openai/gpt-oss-120b"


JUDGE_TOKENIZER_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"


def _render_instruction(step: dict) -> str:
    input_ref = step["input"]
    input_text = (
        input_ref["anchor"]
        if isinstance(input_ref, dict)
        else f"the output of Step {input_ref}"
    )
    return step["instruction"].replace("{input}", input_text)


def ordered_step_evidence(
    semantic_program: dict,
    proof_bank: list[dict],
    evidence_rows: list[dict],
) -> dict[str, dict]:
    """Keep every Plan Step and collect triples under each Retrieve."""

    proofs = {proof["proof_id"]: proof for proof in proof_bank}
    ordered_rows = sorted(evidence_rows, key=lambda item: item["row_id"])
    evidence = {}
    for step in semantic_program["steps"]:
        step_id = step["id"]
        if step["op"] == "intersect":
            evidence[step_id] = {
                "op": "intersect",
                "inputs": list(step["inputs"]),
            }
            continue

        triples = []
        seen = set()
        for row in ordered_rows:
            for proof_id in row["proof_refs_by_step"][step_id]:
                for triple in proofs[proof_id]["proof"]:
                    key = tuple(triple)
                    if key not in seen:
                        seen.add(key)
                        triples.append(triple)
        evidence[step_id] = {
            "op": "retrieve",
            "instruction": _render_instruction(step),
            "input": step["input"],
            "result_role": step["result_role"],
            "triples": triples,
        }
    return evidence


def round_robin_step_evidence(
    evidence: dict[str, dict],
    *,
    fits,
) -> dict[str, dict]:
    """Select globally unique triples fairly across semantic Steps."""

    selected = {}
    for step_id, item in evidence.items():
        if item["op"] == "retrieve":
            selected[step_id] = {
                "op": "retrieve",
                "instruction": item["instruction"],
                "input": item["input"],
                "result_role": item["result_role"],
                "triples": [],
            }
        else:
            selected[step_id] = {
                "op": "intersect",
                "inputs": list(item["inputs"]),
            }
    positions = {
        step_id: 0
        for step_id, item in evidence.items()
        if item["op"] == "retrieve"
    }
    seen = set()
    while True:
        considered = False
        for step_id, item in evidence.items():
            if item["op"] == "intersect":
                continue
            triples = item["triples"]
            position = positions[step_id]
            while position < len(triples) and tuple(triples[position]) in seen:
                position += 1
            positions[step_id] = position
            if position == len(triples):
                continue
            considered = True
            triple = triples[position]
            positions[step_id] += 1
            selected[step_id]["triples"].append(triple)
            if fits(selected):
                seen.add(tuple(triple))
            else:
                selected[step_id]["triples"].pop()
        if not considered:
            break
    return selected


class JudgeTokenCounter:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            JUDGE_TOKENIZER_MODEL,
            revision=JUDGE_TOKENIZER_REVISION,
            local_files_only=True,
            use_fast=True,
        )

    def __call__(self, prompt: str) -> int:
        return len(self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        ))


def run_packing_batch(
    input_path: Path,
    output_path: Path,
    *,
    count_tokens,
) -> dict:
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    errors = Counter()
    for upstream in read_jsonl(input_path):
        status = upstream["status"]
        error_type = upstream["error_type"]
        decision = upstream["semantic_program"].get("decision")
        step_evidence = {}
        packing = {
            "prompt_tokens": 0,
        }
        packing_started = perf_counter() if status == "ok" else None
        if status == "ok":
            decision = decision.strip()
            available_evidence = ordered_step_evidence(
                upstream["semantic_program"],
                proof_bank=upstream["proof_bank"],
                evidence_rows=upstream["evidence_rows"],
            )
            step_evidence = round_robin_step_evidence(
                available_evidence,
                fits=lambda selected: count_tokens(render_judge_prompt(
                    upstream["question"].strip(), decision, selected,
                )) <= JUDGE_INPUT_TOKEN_BUDGET,
            )
            prompt = render_judge_prompt(
                upstream["question"].strip(), decision, step_evidence,
            )
            packing["prompt_tokens"] = count_tokens(prompt)
            if packing["prompt_tokens"] > JUDGE_INPUT_TOKEN_BUDGET:
                raise ValueError("Retrieved triples exceed the Judge token budget")
        packing["packing_seconds"] = (
            perf_counter() - packing_started if packing_started is not None else 0.0
        )

        record = {
            "packing_result_version": PACKING_RESULT_VERSION,
            "index": upstream["index"],
            "example_id": upstream["example_id"],
            "question": upstream["question"],
            "status": status,
            "error_type": error_type,
            "decision": decision,
            "step_evidence": step_evidence,
            "packing": packing,
        }
        records.append(record)
        if status == "error":
            errors[str(error_type or "unknown")] += 1
    write_jsonl(output_path, records, mode="x")
    return {
        "total": len(records),
        "status": dict(sorted(Counter(record["status"] for record in records).items())),
        "errors": dict(sorted(errors.items())),
        "packing_seconds": sum(
            record["packing"]["packing_seconds"] for record in records
        ),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Pack step-organized evidence for evidence-bounded answering.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite packed output: {args.output_path}")
    manifest_path = args.output_path.with_suffix(args.output_path.suffix + ".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite packed manifest: {manifest_path}")
    input_sha256 = sha256_file(args.input_path)
    source_sha256 = source_snapshot_sha256()
    started_at = datetime.now(timezone.utc).isoformat()
    runtime_started = perf_counter()
    token_counter = JudgeTokenCounter()
    summary = run_packing_batch(
        args.input_path,
        args.output_path,
        count_tokens=token_counter,
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
        "runtime_seconds": perf_counter() - runtime_started,
        "configuration": {
            "packing_result_version": PACKING_RESULT_VERSION,
            "input_token_budget": JUDGE_INPUT_TOKEN_BUDGET,
            "judge_tokenizer_model": JUDGE_TOKENIZER_MODEL,
            "judge_tokenizer_revision": JUDGE_TOKENIZER_REVISION,
        },
        "summary": summary,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
