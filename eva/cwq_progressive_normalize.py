"""Normalize one new k=2 CWQ RQ2 output."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.io import load_jsonl, write_jsonl
from .cwq_rq2_normalize_outputs import normalize_condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    registered_ids = {str(row["id"]) for row in load_jsonl(args.cohort)}
    rows = normalize_condition(
        registered_ids,
        load_jsonl(args.input),
        condition="critical",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows, mode="x")


if __name__ == "__main__":
    main()
