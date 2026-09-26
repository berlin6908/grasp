"""Normalize Grasp Final outputs for fixed-cohort RQ2 evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path

from src.io import load_jsonl, write_jsonl

CONDITIONS = ("gplus", "critical", "sham")


def normalize_ours(raw: dict, condition: str) -> dict:
    status = raw["status"]
    answers = raw["answers"] if status == "answered" else []
    if status == "answered" and answers:
        answers = [answer.strip() for answer in answers]
    elif status != "insufficient":
        status, answers = "error", []
    return {
        "id": str(raw["example_id"]),
        "system": "ours",
        "condition": condition,
        "status": status,
        "answers": answers,
    }


def normalize_condition(
    registered_ids: set[str], raw_rows: Iterable[dict], *, condition: str,
) -> list[dict]:
    return [
        normalize_ours(row, condition)
        for row in raw_rows if str(row["example_id"]) in registered_ids
    ]


def normalize_bundle(
    registered_ids: set[str], rows_by_condition: Mapping[str, Iterable[dict]],
) -> list[dict]:
    return [
        row
        for condition in CONDITIONS
        for row in normalize_condition(
            registered_ids, rows_by_condition[condition], condition=condition,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interventions", type=Path, required=True)
    parser.add_argument("--gplus", type=Path, required=True)
    parser.add_argument("--critical", type=Path, required=True)
    parser.add_argument("--sham", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registered_ids = {str(row["id"]) for row in load_jsonl(args.interventions)}
    rows = normalize_bundle(registered_ids, {
        "gplus": load_jsonl(args.gplus),
        "critical": load_jsonl(args.critical),
        "sham": load_jsonl(args.sham),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows, mode="x")


if __name__ == "__main__":
    main()
