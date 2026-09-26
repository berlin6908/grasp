"""Answer matching and the paper's fixed-population metrics."""

from __future__ import annotations

import re
import string
from collections.abc import Callable, Iterable

from src.io import index_rows, require_aligned_ids


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\b(<pad>)\b", " ", text)
    return " ".join(text.split())


def answer_matches(prediction: str, answer: str) -> bool:
    return normalize_answer(answer) in normalize_answer(prediction)


def score_prediction(prediction: str, answers: list[str]) -> dict[str, float]:
    """Score unique gold answers against nonempty prediction lines.

    Matching retains the frozen substring rule, including gold strings that
    normalize to empty text.
    """
    gold = list(dict.fromkeys(answers))
    predictions = [line.strip() for line in prediction.splitlines() if line.strip()]
    normalized_gold = [normalize_answer(answer) for answer in gold]
    normalized_predictions = [normalize_answer(prediction) for prediction in predictions]
    joined_prediction = normalize_answer(" ".join(predictions))
    recalled = sum(answer in joined_prediction for answer in normalized_gold)
    correct = sum(
        any(answer in prediction for answer in normalized_gold)
        for prediction in normalized_predictions
    )
    recall = recalled / len(gold) if gold else 0.0
    precision = correct / len(predictions) if predictions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": recall,
        "hit": float(recalled > 0),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


ReplayGoldQuery = Callable[
    [dict, list[tuple[str, str, str]], list[str], list[tuple[str, str, str]]],
    str,
]


def matched_gold_answers(prediction: str, gold_answers: list[str]) -> list[str]:
    """Return the benchmark answers that produced the row's Hit."""

    return [answer for answer in gold_answers if answer_matches(prediction, answer)]


def summarize_metrics(rows: Iterable[dict]) -> dict:
    """Aggregate standardized row decisions without partition-specific logic."""

    records = list(rows)
    denominator = len(records)
    hit_count = sum(bool(row["hit"]) for row in records)
    certified_count = sum(bool(row["certified_hit"]) for row in records)
    accepted_count = sum(bool(row["accepted"]) for row in records)
    abstention_count = sum(bool(row["abstained"]) for row in records)
    return {
        "hit": {"count": hit_count, "rate": hit_count / denominator},
        "f1": sum(float(row["f1"]) for row in records) / denominator,
        "certified_hit": {
            "count": certified_count,
            "rate": certified_count / denominator,
        },
        "certification_gap": {
            "count": hit_count - certified_count,
            "rate": (hit_count - certified_count) / denominator,
        },
        "complete_witness_rate": {
            "count": certified_count,
            "denominator": accepted_count,
            "rate": certified_count / accepted_count if accepted_count else None,
        },
        "abstention": {
            "count": abstention_count,
            "rate": abstention_count / denominator,
        },
    }


PARTITION_NAMES = {
    "complete_witness": "K",
    "no_witness": "U",
    "K": "K",
    "U": "U",
    "X": "X",
    "unresolved": "unresolved",
}


def _partition(row: dict) -> str:
    value = row["split"] if "split" in row else row["classification"]
    try:
        return PARTITION_NAMES[str(value)]
    except KeyError as error:
        raise ValueError(f"Unknown partition label: {value}") from error


def summarize_paper_metrics(rows: list[dict], partition_rows: list[dict]) -> dict:
    """Report S/I paper measures from the existing per-question decisions."""

    partitions = index_rows(partition_rows, "id", "partition")
    indexed = index_rows(rows, "id", "evaluation")
    require_aligned_ids(set(partitions), set(indexed), "evaluation")
    sufficient = [indexed[key] for key, row in partitions.items() if _partition(row) == "K"]
    insufficient = [indexed[key] for key, row in partitions.items() if _partition(row) == "U"]
    hit = sum(bool(row["hit"]) for row in sufficient)
    supported = sum(bool(row["certified_hit"]) for row in sufficient)
    unknown_s = sum(bool(row["abstained"]) for row in sufficient)
    unknown_i = sum(bool(row["abstained"]) for row in insufficient)
    n_s, n_i = len(sufficient), len(insufficient)
    return {
        "population": {"total": len(rows), "S": n_s, "I": n_i, "excluded": len(rows) - n_s - n_i},
        "counts": {"correct_S": hit, "correct_supported_S": supported, "Unknown_S": unknown_s, "Unknown_I": unknown_i},
        "metrics": {
            "Hit@1": hit / n_s if n_s else None,
            "EvidenceHit@1": supported / n_s if n_s else None,
            "EvidenceGap": (hit - supported) / hit if hit else None,
            "CAR_I": unknown_i / n_i if n_i else None,
            "OAR_S": unknown_s / n_s if n_s else None,
            "EBA": (supported + unknown_i) / (n_s + n_i) if n_s + n_i else None,
        },
    }
