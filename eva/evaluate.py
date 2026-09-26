"""Evaluate Grasp Final records on CWQ, WebQSP, or 2WikiMultiHopQA."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from dataset_split.cwq import load_original_cwq, load_reverse_properties
from dataset_split.webqsp import load_supplemental_mid_names, load_webqsp_questions
from src.io import index_rows, read_jsonl, write_json, write_jsonl
from .evidence import (
    extract_cited_evidence_ids, normalize_triple, packed_evidence_by_id,
    triple_key, unique_triples,
)
from .metrics import (
    ReplayGoldQuery, matched_gold_answers, score_prediction,
    summarize_metrics, summarize_paper_metrics,
)
from .replay import replay_cwq, replay_webqsp, replay_two_wiki


def evaluate_ours_record(
    dataset_row: dict,
    final_row: dict | None,
    replay_gold_query: ReplayGoldQuery,
    *,
    citation_namespace: str = "step",
) -> dict:
    """Score one Ours row from its answer and cited evidence."""

    example_id = str(dataset_row["id"])
    if final_row is None:
        return {
            "id": example_id,
            "status": "missing",
            "top1": "",
            "hit": False,
            "f1": 0.0,
            "accepted": False,
            "abstained": False,
            "cited_evidence_ids": [],
            "invalid_citation_ids": [],
            "selected_triples": [],
            "invalid_selected_triples": [],
            "evidence_replay_status": "no_answer_or_evidence",
            "certified_hit": False,
        }

    status = str(final_row["status"])
    accepted = status == "answered"
    abstained = status == "insufficient"
    answers = [answer.strip() for answer in final_row["answers"]]
    prediction = "\n".join(answers) if accepted else ""
    answer_metrics = score_prediction(prediction, dataset_row["answer"])

    cited_ids: set[str] = set()
    if accepted:
        reason = final_row["reason"]
        if isinstance(reason, str) and reason.strip():
            cited_ids = extract_cited_evidence_ids(reason)
    evidence_by_id = packed_evidence_by_id(
        final_row,
        citation_namespace=citation_namespace,
    )
    valid_ids = sorted(cited_ids & evidence_by_id.keys())
    cited_triples = unique_triples(evidence_by_id[item] for item in valid_ids)
    graph = [normalize_triple(triple) for triple in dataset_row["graph"]]
    graph_keys = {triple_key(triple) for triple in graph}
    unknown = [
        triple for triple in cited_triples if triple_key(triple) not in graph_keys
    ]
    replay_status = "invalid_evidence" if unknown else replay_gold_query(
        dataset_row,
        cited_triples,
        matched_gold_answers(prediction, dataset_row["answer"]),
        graph,
    )
    hit = answer_metrics["hit"] > 0.0
    return {
        "id": example_id,
        "status": (
            "answer" if accepted else "abstention" if abstained else "error"
        ),
        "top1": answers[0] if accepted and answers else "",
        "hit": hit,
        "f1": answer_metrics["f1"],
        "accepted": accepted,
        "abstained": abstained,
        "cited_evidence_ids": valid_ids,
        "invalid_citation_ids": sorted(cited_ids - evidence_by_id.keys()),
        "selected_triples": [list(triple) for triple in cited_triples],
        "invalid_selected_triples": [list(triple) for triple in unknown],
        "evidence_replay_status": replay_status,
        "certified_hit": hit and replay_status == "success",
    }


def evaluate_rows(
    dataset: list[dict],
    final_rows: list[dict],
    partition_rows: list[dict],
    replay: ReplayGoldQuery,
    *,
    citation_namespace: str = "step",
) -> tuple[dict, list[dict]]:
    if not dataset:
        raise ValueError("Dataset must not be empty")
    indexed_dataset = index_rows(dataset, "id", "dataset")
    predictions = index_rows(final_rows, "example_id", "Final")
    if predictions.keys() - indexed_dataset.keys():
        raise ValueError("Final contains IDs outside the evaluation dataset")
    # Final already carries the packed step_evidence used by the answer generator.
    details = [
        evaluate_ours_record(
            row, predictions.get(key), replay,
            citation_namespace=citation_namespace,
        )
        for key, row in indexed_dataset.items()
    ]
    summary = summarize_paper_metrics(details, partition_rows)
    summary["internal_full_population"] = summarize_metrics(details)
    return summary, details


def _build_replay(args: argparse.Namespace, partitions: list[dict]) -> ReplayGoldQuery:
    if args.benchmark == "2wiki":
        return partial(replay_two_wiki, labels=index_rows(partitions, "id", "2Wiki labels"))
    if args.reverse_properties is None:
        raise ValueError("CWQ and WebQSP require --reverse-properties")
    reverse_properties = load_reverse_properties(args.reverse_properties)
    if args.benchmark == "cwq":
        if args.original_cwq is None or args.witness_details is None:
            raise ValueError("CWQ requires --original-cwq and --witness-details")
        return partial(
            replay_cwq,
            original_by_id=load_original_cwq(args.original_cwq),
            witness_by_id=index_rows(read_jsonl(args.witness_details), "id", "CWQ constraints"),
            reverse_properties=reverse_properties,
        )
    if args.official_webqsp is None or args.supplemental_mid_names is None:
        raise ValueError("WebQSP requires --official-webqsp and --supplemental-mid-names")
    return partial(
        replay_webqsp,
        official_questions=load_webqsp_questions(args.official_webqsp),
        supplemental_mid_names=load_supplemental_mid_names(args.supplemental_mid_names),
        reverse_properties=reverse_properties,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("cwq", "webqsp", "2wiki"), required=True)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--final", type=Path, nargs="+", required=True)
    parser.add_argument("--partition-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flat-evidence", action="store_true")
    parser.add_argument("--original-cwq", type=Path)
    parser.add_argument("--witness-details", type=Path)
    parser.add_argument("--reverse-properties", type=Path)
    parser.add_argument("--official-webqsp", type=Path)
    parser.add_argument("--supplemental-mid-names", type=Path)
    args = parser.parse_args()
    dataset = [row for path in args.dataset for row in read_jsonl(path)]
    if args.benchmark == "2wiki":
        dataset = [
            {**row, "answer": row.get("ground_truth") or [row["answer"]]}
            for row in dataset
        ]
    partitions = list(read_jsonl(args.partition_index))
    predictions = [row for path in args.final for row in read_jsonl(path)]
    summary, details = evaluate_rows(
        dataset, predictions, partitions, _build_replay(args, partitions),
        citation_namespace="flat" if args.flat_evidence else "step",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "details.jsonl", details)
    print(json.dumps(
        {key: summary[key] for key in ("population", "counts", "metrics")},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
