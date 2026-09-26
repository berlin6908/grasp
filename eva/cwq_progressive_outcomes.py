"""Evaluate CWQ RQ2 answer persistence over k=0,1,2."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from math import sqrt
from pathlib import Path

from src.io import load_jsonl, write_json, write_jsonl
from .cwq_rq2_outcomes import (
    _count_rate,
    _transition,
    _transition_summary,
    paper_transition_metrics,
)


def _index(
    rows: Iterable[dict],
    *,
    condition: str,
    expected_ids: set[str],
) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        if row["system"] != "ours" or row.get("condition") != condition:
            continue
        example_id = str(row["id"])
        if example_id not in expected_ids:
            continue
        if example_id in indexed:
            raise ValueError(f"duplicate Grasp {condition} output: {example_id}")
        indexed[example_id] = row
    return indexed


def _wilson(count: int, denominator: int) -> dict:
    z = 1.959963984540054
    proportion = count / denominator
    scale = 1 + z * z / denominator
    center = (proportion + z * z / (2 * denominator)) / scale
    margin = (
        z
        * sqrt(
            proportion * (1 - proportion) / denominator
            + z * z / (4 * denominator * denominator)
        )
        / scale
    )
    return {
        "method": "Wilson",
        "level": 0.95,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def _level_summary(rows: list[dict], level: str) -> dict:
    denominator = len(rows)
    transitions = _transition_summary(rows, f"k{level}")
    same = transitions["same_answer"]["count"]
    abstention = transitions["abstention"]["count"]
    accepted = same + transitions["other_answer"]["count"]
    return {
        "transitions": transitions,
        "apr": {**transitions["same_answer"], "confidence_interval": _wilson(same, denominator)},
        "car": _count_rate(abstention, denominator),
        "far": _count_rate(accepted, denominator),
    }


def evaluate_progressive(
    cohort: Iterable[dict],
    k1_rows: Iterable[dict],
    k2_rows: Iterable[dict],
) -> dict:
    """Classify all three levels on the same pre-inference cohort."""

    cohort_rows = list(cohort)
    if not cohort_rows:
        raise ValueError("progressive cohort must not be empty")
    ids = [str(row["id"]) for row in cohort_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("progressive cohort ids must be unique")
    expected = set(ids)

    k1 = _index(k1_rows, condition="critical", expected_ids=expected)
    k2 = _index(k2_rows, condition="critical", expected_ids=expected)

    details = []
    for cohort_row in cohort_rows:
        example_id = str(cohort_row["id"])
        baseline_top1 = cohort_row["system_top1"]["ours"]
        k0 = (
            {
                "status": "answered",
                "top1": baseline_top1,
                "transition": "same_answer",
            }
            if baseline_top1 is not None
            else {"status": "error", "top1": "", "transition": "error"}
        )
        details.append(
            {
                "id": example_id,
                "system": "ours",
                "k0": k0,
                "k1": _transition(k1.get(example_id), baseline_top1, True),
                "k2": _transition(k2.get(example_id), baseline_top1, True),
            }
        )

    return {
        "system": "ours",
        "denominator": len(details),
        "levels": {str(level): _level_summary(details, str(level)) for level in range(3)},
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--k1", required=True, type=Path)
    parser.add_argument("--k2", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite outcomes: {args.output_dir}")
    result = evaluate_progressive(
        load_jsonl(args.cohort),
        load_jsonl(args.k1),
        load_jsonl(args.k2),
    )
    args.output_dir.mkdir(parents=True)
    details = result.pop("details")
    write_jsonl(args.output_dir / "details.jsonl", details)
    write_json(args.output_dir / "summary.json", result)
    print(json.dumps({
        "system": "Grasp", "N": result["denominator"],
        "levels": {level: paper_transition_metrics(summary["transitions"])
                   for level, summary in result["levels"].items()},
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
