"""Construct validated CWQ missing-condition and sham graph pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from math import ceil
from pathlib import Path
from typing import Callable

from dataset_split.cwq import (
    QueryPlan,
    canonical_text,
    candidate_matches,
    evaluate_query,
    load_original_cwq,
    load_reverse_properties,
    parse_query,
)
from src.io import index_rows, read_jsonl, write_jsonl
from eva.evidence import (
    graph_conditioned_answers,
    normalize_triple,
)


Triple = tuple[str, str, str]


def _without_indices(graph: list[Triple], graph_indices: set[int]) -> list[Triple]:
    return [
        triple
        for index, triple in enumerate(graph, start=1)
        if index not in graph_indices
    ]


def _answer_nodes_present(graph: list[Triple], answers: list[str]) -> bool:
    nodes = {
        canonical_text(node)
        for head, _relation, tail in graph
        for node in (head, tail)
    }
    return all(canonical_text(answer) in nodes for answer in answers)


def select_missing_condition_pair(
    graph: list[Triple],
    *,
    plan: QueryPlan,
    reverse_properties: dict[str, str],
    protected_nodes: list[str],
    witness_exists: Callable[[list[Triple]], bool],
) -> dict | None:
    """Remove every graph match for one required pattern in each SPARQL branch."""

    if not witness_exists(graph):
        return None

    branch_options = []
    query_match_indices: set[int] = set()
    graph_rows = [list(triple) for triple in graph]
    for branch_number, branch in enumerate(plan.branches, start=1):
        matches = candidate_matches(branch, graph_rows, reverse_properties)
        query_match_indices.update(match.graph_index for match in matches)
        options = []
        for pattern in branch:
            indices = sorted(
                {
                    match.graph_index
                    for match in matches
                    if pattern.index in match.covered
                }
            )
            if indices:
                options.append(
                    (
                        {
                            "branch": branch_number,
                            "pattern_index": pattern.index,
                            "subject": pattern.subject,
                            "relation": pattern.relation,
                            "object": pattern.object,
                        },
                        indices,
                    )
                )
        if not options:
            return None
        branch_options.append(options)

    combinations = []
    for selected in product(*branch_options):
        critical_indices = sorted(
            {index for _condition, indices in selected for index in indices}
        )
        combinations.append(
            (
                len(critical_indices),
                tuple(condition["pattern_index"] for condition, _indices in selected),
                [condition for condition, _indices in selected],
                critical_indices,
            )
        )

    chosen = None
    for _count, _pattern_order, conditions, critical_indices in sorted(combinations):
        critical_graph = _without_indices(graph, set(critical_indices))
        if not _answer_nodes_present(critical_graph, protected_nodes):
            continue
        if not witness_exists(critical_graph):
            chosen = (conditions, critical_indices)
            break
    if chosen is None:
        return None

    conditions, critical_indices = chosen
    sham = _select_sham_edges(
        graph,
        query_match_indices=query_match_indices,
        removal_count=len(critical_indices),
        protected_nodes=protected_nodes,
        witness_exists=witness_exists,
    )

    return {
        "intervention_mode": "gold_query_pattern_ablation",
        "target_conditions": conditions,
        "critical_graph_indices": critical_indices,
        "critical_triples": [graph[index - 1] for index in critical_indices],
        **sham,
    }


def greedy_witness_cut(
    graph: list[Triple],
    *,
    protected_nodes: list[str],
    evaluate_witness: Callable[[list[Triple]], dict],
) -> list[int] | None:
    """Greedily hit returned witnesses, then remove redundant deleted edges."""

    removed: set[int] = set()

    def evaluate(indices: set[int]) -> tuple[dict, list[int]]:
        active = [index for index in range(1, len(graph) + 1) if index not in indices]
        result = evaluate_witness([graph[index - 1] for index in active])
        return result, active

    while True:
        result, active = evaluate(removed)
        if not result["has_complete_witness"]:
            break
        local_indices = {
            *[int(index) for index in result["witness_graph_indices"]],
            *[int(index) for index in result["inverse_witness_graph_indices"]],
        }
        original_indices = sorted(
            active[index - 1]
            for index in local_indices
            if 1 <= index <= len(active)
        )
        chosen = None
        for index in original_indices:
            candidate_removed = {*removed, index}
            candidate_graph = _without_indices(graph, candidate_removed)
            if _answer_nodes_present(candidate_graph, protected_nodes):
                chosen = index
                break
        if chosen is None:
            return None
        removed.add(chosen)
        if len(removed) > len(graph):
            return None

    for index in sorted(removed):
        candidate_removed = removed - {index}
        candidate_graph = _without_indices(graph, candidate_removed)
        if not _answer_nodes_present(candidate_graph, protected_nodes):
            continue
        result, _active = evaluate(candidate_removed)
        if not result["has_complete_witness"]:
            removed = candidate_removed
    return sorted(removed)


def select_non_query_sham(
    graph: list[Triple],
    *,
    plan: QueryPlan,
    reverse_properties: dict[str, str],
    removal_count: int,
    protected_nodes: list[str],
    witness_exists: Callable[[list[Triple]], bool],
) -> dict:
    """Find an equal-size negative control outside all gold-query matches."""

    graph_rows = [list(triple) for triple in graph]
    query_match_indices = {
        match.graph_index
        for branch in plan.branches
        for match in candidate_matches(branch, graph_rows, reverse_properties)
    }
    return _select_sham_edges(
        graph,
        query_match_indices=query_match_indices,
        removal_count=removal_count,
        protected_nodes=protected_nodes,
        witness_exists=witness_exists,
    )


def _select_sham_edges(
    graph: list[Triple],
    *,
    query_match_indices: set[int],
    removal_count: int,
    protected_nodes: list[str],
    witness_exists: Callable[[list[Triple]], bool],
) -> dict:
    protected_keys = {canonical_text(node) for node in protected_nodes}
    by_relation: dict[str, list[int]] = {}
    for index, triple in enumerate(graph, start=1):
        if index not in query_match_indices:
            by_relation.setdefault(triple[1], []).append(index)
    options = sorted(
        (
            sum(
                canonical_text(graph[index - 1][0]) in protected_keys
                or canonical_text(graph[index - 1][2]) in protected_keys
                for index in indices[:removal_count]
            ),
            abs(len(indices) - removal_count),
            relation,
            indices,
        )
        for relation, indices in by_relation.items()
        if len(indices) >= removal_count
    )
    for _incidence, _distance, _relation, indices in options:
        candidate_indices = indices[:removal_count]
        candidate_graph = _without_indices(graph, set(candidate_indices))
        if _answer_nodes_present(candidate_graph, protected_nodes) and witness_exists(
            candidate_graph
        ):
            return {
                "sham_graph_indices": candidate_indices,
                "sham_triples": [graph[index - 1] for index in candidate_indices],
                "sham_match": "same_cardinality_non_query_relation",
            }
    candidates = sorted(set(range(1, len(graph) + 1)) - query_match_indices)
    if len(candidates) >= removal_count:
        candidate_indices = candidates[:removal_count]
        candidate_graph = _without_indices(graph, set(candidate_indices))
        if _answer_nodes_present(candidate_graph, protected_nodes) and witness_exists(
            candidate_graph
        ):
            return {
                "sham_graph_indices": candidate_indices,
                "sham_triples": [graph[index - 1] for index in candidate_indices],
                "sham_match": "same_cardinality_mixed_non_query_edges",
            }
    return {
        "sham_graph_indices": [],
        "sham_triples": [],
        "sham_match": "unavailable",
    }


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def build_interventions(
    *,
    k_path: Path,
    common_pool_path: Path,
    original_cwq_path: Path,
    witness_details_path: Path,
    reverse_properties_path: Path,
    output_dir: Path,
) -> dict:
    k_rows = list(read_jsonl(k_path))
    k_by_id = index_rows(k_rows, "id", "CWQ-K")
    common_rows = list(read_jsonl(common_pool_path))
    common_by_id = index_rows(common_rows, "id", "common-pool")
    if any(not row.get("common_certified_hit") for row in common_rows):
        raise ValueError("common pool contains a row outside the correct-and-supported RQ2 cohort")
    unknown_ids = set(common_by_id) - set(k_by_id)
    if unknown_ids:
        raise ValueError(f"common-pool ids absent from CWQ-K: {sorted(unknown_ids)[:3]}")

    original_by_id = load_original_cwq(original_cwq_path)
    witness_by_id = index_rows(read_jsonl(witness_details_path), "id", "witness")
    reverse_properties = load_reverse_properties(reverse_properties_path)

    manifests = []
    gplus_rows = []
    gminus_rows = []
    gsham_rows = []
    failures = Counter()

    for dataset_row in k_rows:
        example_id = str(dataset_row["id"])
        pool_row = common_by_id.get(example_id)
        if pool_row is None:
            continue

        graph = [normalize_triple(triple) for triple in dataset_row["graph"]]
        graph_gold = graph_conditioned_answers(dataset_row["answer"], graph)
        original = original_by_id[example_id]
        witness = witness_by_id[example_id]
        plan = parse_query(original["sparql"])
        mid_bindings = {
            str(mid).strip(): str(node).strip()
            for mid, node in witness["mid_bindings"].items()
        }

        def evaluate_witness(candidate_graph: list[Triple]) -> dict:
            return evaluate_query(
                plan,
                [list(triple) for triple in candidate_graph],
                mid_bindings,
                graph_gold,
                reverse_properties,
            )

        def witness_exists(candidate_graph: list[Triple]) -> bool:
            result = evaluate_witness(candidate_graph)
            return bool(result["has_complete_witness"])

        protected_nodes = graph_gold
        try:
            pair = select_missing_condition_pair(
                graph,
                plan=plan,
                reverse_properties=reverse_properties,
                protected_nodes=protected_nodes,
                witness_exists=witness_exists,
            )
        except ValueError:
            failures["replay_unresolved"] += 1
            continue
        if pair is None:
            try:
                critical_indices = greedy_witness_cut(
                    graph,
                    protected_nodes=protected_nodes,
                    evaluate_witness=evaluate_witness,
                )
            except ValueError:
                failures["fallback_replay_unresolved"] += 1
                continue
            if not critical_indices:
                failures["no_validated_missing_condition_intervention"] += 1
                continue
            sham = select_non_query_sham(
                graph,
                plan=plan,
                reverse_properties=reverse_properties,
                removal_count=len(critical_indices),
                protected_nodes=protected_nodes,
                witness_exists=witness_exists,
            )
            pair = {
                "intervention_mode": "binding_specific_witness_cut",
                "target_conditions": [],
                "critical_graph_indices": critical_indices,
                "critical_triples": [
                    graph[index - 1] for index in critical_indices
                ],
                **sham,
            }

        critical_indices = [int(index) for index in pair["critical_graph_indices"]]
        sham_indices = [int(index) for index in pair["sham_graph_indices"]]
        gminus_graph = _without_indices(graph, set(critical_indices))
        gsham_graph = (
            _without_indices(graph, set(sham_indices)) if sham_indices else None
        )
        original_type = str(original.get("compositionality_type") or "unknown")
        manifest = {
            "id": example_id,
            "question": dataset_row.get("question"),
            "compositionality_type": original_type,
            "graph_conditioned_gold": graph_gold,
            "system_top1": {"ours": pool_row["systems"]["ours"]["top1"]},
            "intervention_type": pair["intervention_mode"],
            "target_conditions": pair["target_conditions"],
            "critical_graph_indices": critical_indices,
            "critical_triples": [list(triple) for triple in pair["critical_triples"]],
            "sham_match": pair["sham_match"],
            "sham_graph_indices": sham_indices,
            "sham_triples": [list(triple) for triple in pair["sham_triples"]],
            "original_graph_triple_count": len(graph),
            "critical_removed_count": len(critical_indices),
            "critical_removed_fraction": len(critical_indices) / len(graph),
            "validations": {
                "gplus_has_complete_witness": True,
                "gminus_has_complete_witness": False,
                "gsham_has_complete_witness": True if sham_indices else None,
                "gold_answer_nodes_preserved": True,
            },
        }
        manifests.append(manifest)
        gplus_rows.append(dataset_row)
        gminus_rows.append({**dataset_row, "graph": [list(triple) for triple in gminus_graph]})
        if gsham_graph is not None:
            gsham_rows.append(
                {**dataset_row, "graph": [list(triple) for triple in gsham_graph]}
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "interventions.jsonl", manifests)
    write_jsonl(output_dir / "gplus.jsonl", gplus_rows)
    write_jsonl(output_dir / "gminus.jsonl", gminus_rows)
    write_jsonl(output_dir / "gsham.jsonl", gsham_rows)

    type_counts = Counter(row["compositionality_type"] for row in manifests)
    intervention_counts = Counter(row["intervention_type"] for row in manifests)
    sham_counts = Counter(row["sham_match"] for row in manifests)
    removal_counts = [float(row["critical_removed_count"]) for row in manifests]
    removal_fractions = [
        100.0 * float(row["critical_removed_fraction"]) for row in manifests
    ]
    summary = {
        "protocol": {
            "name": "cwq-v3-missing-condition-intervention-v1",
            "source_pool": "provided frozen intervention cohort",
            "critical_intervention": (
                "remove every graph match for one required gold-query pattern per SPARQL "
                "branch, preserve every graph-conditioned gold answer entity, and "
                "require full-graph gold-query replay to fail"
            ),
            "sham_intervention": (
                "optional negative control that removes the same number of edges which "
                "match no gold-query pattern, preferentially from one unrelated relation, "
                "while requiring full-graph gold-query replay to remain successful"
            ),
        },
        "source_common_certified_rows": len(common_rows),
        "validated_interventions": len(manifests),
        "validated_sham_controls": sum(
            int(bool(row["sham_graph_indices"])) for row in manifests
        ),
        "by_compositionality_type": dict(sorted(type_counts.items())),
        "by_intervention_type": dict(sorted(intervention_counts.items())),
        "by_sham_match": dict(sorted(sham_counts.items())),
        "critical_removal_count": {
            "min": min(removal_counts, default=0),
            "median": _nearest_rank(removal_counts, 0.5),
            "p75": _nearest_rank(removal_counts, 0.75),
            "p95": _nearest_rank(removal_counts, 0.95),
            "max": max(removal_counts, default=0),
            "mean": sum(removal_counts) / len(removal_counts) if removal_counts else 0.0,
        },
        "critical_removal_percent_of_graph": {
            "median": _nearest_rank(removal_fractions, 0.5),
            "p75": _nearest_rank(removal_fractions, 0.75),
            "p95": _nearest_rank(removal_fractions, 0.95),
            "max": max(removal_fractions, default=0),
            "at_most_1_percent": sum(value <= 1.0 for value in removal_fractions),
            "at_most_5_percent": sum(value <= 5.0 for value in removal_fractions),
        },
        "excluded": dict(sorted(failures.items())),
        "sources": {
            "cwq_k": str(k_path.resolve()),
            "common_pool": str(common_pool_path.resolve()),
            "original_cwq": str(original_cwq_path.resolve()),
            "witness_details": str(witness_details_path.resolve()),
            "reverse_properties": str(reverse_properties_path.resolve()),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", required=True, type=Path)
    parser.add_argument("--common-pool", required=True, type=Path)
    parser.add_argument("--original-cwq", required=True, type=Path)
    parser.add_argument("--witness-details", required=True, type=Path)
    parser.add_argument("--reverse-properties", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_interventions(
        k_path=args.k,
        common_pool_path=args.common_pool,
        original_cwq_path=args.original_cwq,
        witness_details_path=args.witness_details,
        reverse_properties_path=args.reverse_properties,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
