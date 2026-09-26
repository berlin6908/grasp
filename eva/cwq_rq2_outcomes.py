"""Evaluate fixed-population CWQ RQ2 answer transitions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from math import comb, sqrt
from pathlib import Path
from statistics import NormalDist

from src.io import index_rows, load_jsonl, require_aligned_ids, write_json, write_jsonl
from .metrics import normalize_answer


EXPECTED_DENOMINATORS = {
    "missing_condition": 295,
    "binding_conflict": 120,
}
CONDITIONS = ("gplus", "critical", "sham")
TRANSITIONS = ("same_answer", "other_answer", "abstention", "error")
OUTPUT_STATUSES = {"answered", "insufficient", "error"}


def load_preflight(path: Path) -> dict[str, bool]:
    """Read the Grasp graph preflight produced by cwq_rq2_inputs."""
    rows = index_rows(load_jsonl(path), "id", "preflight")
    return {example_id: row["status"] == "ok" for example_id, row in rows.items()}


def _index_predictions(
    rows: Iterable[dict], expected_ids: set[str]
) -> dict[tuple[str, str], dict]:
    indexed: dict[tuple[str, str], dict] = {}
    observed_ids: set[str] = set()
    for row in rows:
        if row["system"] != "ours":
            continue
        example_id = str(row["id"])
        condition = row.get("condition")
        status = row.get("status")
        answers = row.get("answers")
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition for {example_id}: {condition!r}")
        if status not in OUTPUT_STATUSES:
            raise ValueError(f"unknown output status for {example_id}: {status!r}")
        if not isinstance(answers, list) or any(not isinstance(answer, str) for answer in answers):
            raise ValueError(f"answers must be a list of strings for {example_id}")
        key = (example_id, condition)
        if key in indexed:
            raise ValueError(f"duplicate prediction row: {key}")
        indexed[key] = row
        observed_ids.add(example_id)
    extra = observed_ids - expected_ids
    if extra:
        raise ValueError(f"prediction ids outside the registered population: {sorted(extra)[:3]}")
    return indexed


def _output_view(row: dict | None) -> dict:
    if row is None:
        return {"status": "error", "top1": "", "reason": "missing_output"}
    if row["status"] == "insufficient":
        return {"status": "insufficient", "top1": ""}
    if row["status"] == "error":
        return {"status": "error", "top1": ""}
    top1 = row["answers"][0] if row["answers"] else ""
    if not normalize_answer(top1):
        return {"status": "error", "top1": top1, "reason": "empty_answered_top1"}
    return {"status": "answered", "top1": top1}


def _transition(
    row: dict | None,
    baseline_top1: str | None,
    preflight_passed: bool,
) -> dict:
    if not preflight_passed:
        return {
            "status": "error",
            "top1": "",
            "transition": "error",
            "reason": "preflight_failure",
        }
    view = _output_view(row)
    if view["status"] == "insufficient":
        return {**view, "transition": "abstention"}
    if baseline_top1 is None:
        return {**view, "transition": "error", "reason": "invalid_gplus_baseline"}
    if view["status"] != "answered":
        return {**view, "transition": "error"}
    transition = (
        "same_answer"
        if normalize_answer(view["top1"]) == normalize_answer(baseline_top1)
        else "other_answer"
    )
    return {**view, "transition": transition}


def _count_rate(count: int, denominator: int) -> dict:
    return {"count": count, "rate": count / denominator}


def _exact_mcnemar_p(critical_only: int, sham_only: int) -> float:
    discordant = critical_only + sham_only
    if not discordant:
        return 1.0
    smaller = min(critical_only, sham_only)
    lower_tail = sum(comb(discordant, value) for value in range(smaller + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def paired_binary_contrast(
    critical: Iterable[bool], sham: Iterable[bool], *, direction: str
) -> dict:
    """Return a paired risk difference, paired interval, and exact McNemar test."""

    critical_values = list(critical)
    sham_values = list(sham)
    if len(critical_values) != len(sham_values) or not critical_values:
        raise ValueError("paired binary outcomes must be non-empty and aligned")
    critical_only = sum(c and not s for c, s in zip(critical_values, sham_values))
    sham_only = sum(s and not c for c, s in zip(critical_values, sham_values))
    both = sum(c and s for c, s in zip(critical_values, sham_values))
    neither = len(critical_values) - critical_only - sham_only - both
    discordant = critical_only + sham_only
    if direction == "sham_minus_critical":
        estimate = (sham_only - critical_only) / len(critical_values)
    elif direction == "critical_minus_sham":
        estimate = (critical_only - sham_only) / len(critical_values)
    else:
        raise ValueError(f"unknown contrast direction: {direction}")

    denominator = len(critical_values)
    adjusted_critical = (critical_only + 1) / (denominator + 2)
    adjusted_sham = (sham_only + 1) / (denominator + 2)
    if direction == "sham_minus_critical":
        adjusted_estimate = adjusted_sham - adjusted_critical
    else:
        adjusted_estimate = adjusted_critical - adjusted_sham
    standard_error = sqrt(
        (
            adjusted_critical
            + adjusted_sham
            - adjusted_estimate**2
        )
        / (denominator + 2)
    )
    margin = NormalDist().inv_cdf(0.975) * standard_error
    exact_p = _exact_mcnemar_p(critical_only, sham_only)
    return {
        "direction": direction,
        "estimate": estimate,
        "critical_only": critical_only,
        "sham_only": sham_only,
        "both": both,
        "neither": neither,
        "discordant": discordant,
        "confidence_interval": {
            "method": "Bonett-Price adjusted-Wald",
            "level": 0.95,
            "adjusted_estimate": adjusted_estimate,
            "lower": max(-1.0, adjusted_estimate - margin),
            "upper": min(1.0, adjusted_estimate + margin),
        },
        "exact_mcnemar_p": exact_p,
    }


def _transition_summary(details: list[dict], condition: str) -> dict:
    denominator = len(details)
    counts = Counter(row[condition]["transition"] for row in details)
    return {
        transition: _count_rate(counts[transition], denominator)
        for transition in TRANSITIONS
    }


def paper_transition_metrics(transitions: dict) -> dict:
    """Present recorded transition counts and rates using paper labels."""
    return {
        "APR": transitions["same_answer"],
        "ATR_Same": transitions["same_answer"],
        "ATR_Other": transitions["other_answer"],
        "ATR_Unknown": transitions["abstention"],
        "Error": transitions["error"],
    }


def summarize_transitions(details: Iterable[dict]) -> dict:
    """Summarize already classified paired rows on their current denominator."""

    rows = list(details)
    if not rows:
        raise ValueError("classified RQ2 rows must not be empty")
    critical_transitions = _transition_summary(rows, "critical")
    sham_transitions = _transition_summary(rows, "sham")
    return {
        "critical": {
            "transitions": critical_transitions,
            "apr": critical_transitions["same_answer"],
            "car": critical_transitions["abstention"],
            "far": _count_rate(
                critical_transitions["same_answer"]["count"]
                + critical_transitions["other_answer"]["count"],
                len(rows),
            ),
        },
        "sham": {
            "transitions": sham_transitions,
            "persistence": sham_transitions["same_answer"],
            "abstention": sham_transitions["abstention"],
        },
        "paired_contrasts": {
            "persistence": paired_binary_contrast(
                [
                    row["critical"]["transition"] == "same_answer"
                    for row in rows
                ],
                [row["sham"]["transition"] == "same_answer" for row in rows],
                direction="sham_minus_critical",
            ),
            "abstention": paired_binary_contrast(
                [
                    row["critical"]["transition"] == "abstention"
                    for row in rows
                ],
                [row["sham"]["transition"] == "abstention" for row in rows],
                direction="critical_minus_sham",
            ),
        },
        "confidence_intervals": {
            "method": "Bonett-Price adjusted-Wald paired risk difference",
            "level": 0.95,
            "multiplicity": "marginal_unadjusted",
        },
    }


def evaluate_rq2(
    interventions: Iterable[dict],
    preflight_pass: dict[str, bool],
    prediction_rows: Iterable[dict],
    *,
    family: str,
) -> dict:
    """Classify Grasp and one intervention family on its registered denominator."""

    if family not in EXPECTED_DENOMINATORS:
        raise ValueError(f"unknown intervention family: {family}")
    interventions_by_id = index_rows(interventions, "id", "intervention")
    intervention_order = list(interventions_by_id)
    denominator = EXPECTED_DENOMINATORS[family]
    if len(intervention_order) != denominator:
        raise ValueError(
            f"{family} requires {denominator} registered rows, got {len(intervention_order)}"
        )
    expected_ids = set(interventions_by_id)

    require_aligned_ids(expected_ids, set(preflight_pass), "preflight")
    predictions = _index_predictions(prediction_rows, expected_ids)

    details: list[dict] = []
    for example_id in intervention_order:
        gplus = _output_view(predictions.get((example_id, "gplus")))
        baseline_top1 = gplus["top1"] if gplus["status"] == "answered" else None
        passed = preflight_pass[example_id]
        critical = _transition(
            predictions.get((example_id, "critical")), baseline_top1, passed
        )
        sham = _transition(predictions.get((example_id, "sham")), baseline_top1, passed)
        details.append(
            {
                "id": example_id,
                "system": "ours",
                "family": family,
                "preflight_status": "pass" if passed else "error",
                "gplus": gplus,
                "critical": critical,
                "sham": sham,
            }
        )

    summary = {
        "system": "ours",
        "family": family,
        "denominator": denominator,
        "preflight": {
            "pass": sum(preflight_pass.values()),
            "error": denominator - sum(preflight_pass.values()),
        },
        **summarize_transitions(details),
    }
    return {"summary": summary, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=EXPECTED_DENOMINATORS, required=True)
    parser.add_argument("--interventions", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.output_dir / "summary.json"
    details_path = args.output_dir / "details.jsonl"
    if summary_path.exists() or details_path.exists():
        raise FileExistsError(f"refusing to overwrite RQ2 outcomes: {args.output_dir}")
    result = evaluate_rq2(
        load_jsonl(args.interventions),
        load_preflight(args.preflight),
        load_jsonl(args.predictions),
        family=args.family,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(details_path, result["details"])
    write_json(summary_path, result["summary"])
    summary = result["summary"]
    print(json.dumps({
        "system": "Grasp",
        "family": {"missing_condition": "Condition loss", "binding_conflict": "Binding mismatch"}[args.family],
        "N": summary["denominator"],
        "metrics": paper_transition_metrics(summary["critical"]["transitions"]),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
