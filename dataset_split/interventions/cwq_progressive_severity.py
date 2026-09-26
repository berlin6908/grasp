"""Construct the minimal CWQ RQ2 progressive-severity intervention."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Callable

from dataset_split.cwq import (
    QueryPlan,
    candidate_matches,
    evaluate_query,
    load_original_cwq,
    load_reverse_properties,
    parse_query,
)
from src.io import read_jsonl, write_jsonl
from .cwq_witness_breaking import Triple, _answer_nodes_present, _without_indices
from eva.evidence import (
    graph_conditioned_answers,
    normalize_triple,
)


def _condition(pattern) -> dict:
    return {
        "branch": 1,
        "pattern_index": pattern.index,
        "subject": pattern.subject,
        "relation": pattern.relation,
        "object": pattern.object,
    }


def select_progressive_sequence(
    graph: list[Triple],
    *,
    plan: QueryPlan,
    reverse_properties: dict[str, str],
    protected_nodes: list[str],
    witness_exists: Callable[[list[Triple]], bool],
    first_condition: dict,
    first_critical_indices: list[int],
) -> dict | None:
    """Extend a validated k=1 deletion with one more required condition."""

    if len(plan.branches) != 1:
        return None

    branch = plan.branches[0]
    matches = candidate_matches(branch, [list(triple) for triple in graph], reverse_properties)
    matches_by_pattern = {
        pattern.index: sorted(
            {match.graph_index for match in matches if pattern.index in match.covered}
        )
        for pattern in branch
    }
    matches_by_pattern = {
        index: indices for index, indices in matches_by_pattern.items() if indices
    }
    first_index = int(first_condition["pattern_index"])
    if first_index not in matches_by_pattern or len(matches_by_pattern) < 2:
        return None

    first_critical = sorted({int(index) for index in first_critical_indices})
    if witness_exists(_without_indices(graph, set(first_critical))):
        return None

    patterns = {pattern.index: pattern for pattern in branch}
    candidates = []
    for second in sorted(set(matches_by_pattern) - {first_index}):
        added = sorted(set(matches_by_pattern[second]) - set(first_critical))
        if not added:
            continue

        critical_indices = sorted(set(first_critical) | set(added))
        critical_graph = _without_indices(graph, set(critical_indices))
        if witness_exists(critical_graph):
            continue

        answer_nodes_preserved = _answer_nodes_present(critical_graph, protected_nodes)
        candidates.append(
            (
                not answer_nodes_preserved,
                len(critical_indices),
                second,
                critical_indices,
                added,
                answer_nodes_preserved,
            )
        )

    if not candidates:
        return None

    (
        _prefer_answer_retention,
        _removed_count,
        second,
        critical_indices,
        added,
        answer_nodes_preserved,
    ) = min(candidates)
    return {
        "conditions": [first_condition, _condition(patterns[second])],
        "levels": {
            1: {"critical_graph_indices": first_critical},
            2: {
                "critical_graph_indices": critical_indices,
                "added_graph_indices": added,
                "gold_answer_nodes_preserved": answer_nodes_preserved,
            },
        },
    }


def build_progressive_bundle(
    *,
    gplus_path: Path,
    missing_interventions_path: Path,
    original_cwq_path: Path,
    witness_details_path: Path,
    reverse_properties_path: Path,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite progressive bundle: {output_dir}")

    gplus_rows = list(read_jsonl(gplus_path))
    interventions = {
        str(row["id"]): row for row in read_jsonl(missing_interventions_path)
    }
    witness_rows = {str(row["id"]): row for row in read_jsonl(witness_details_path)}
    original_rows = load_original_cwq(original_cwq_path)
    reverse_properties = load_reverse_properties(reverse_properties_path)
    excluded = Counter()
    cohort_rows = []
    k2_gminus_rows = []
    k2_interventions = []

    for dataset_row in gplus_rows:
        example_id = str(dataset_row["id"])
        intervention = interventions[example_id]
        plan = parse_query(original_rows[example_id]["sparql"])
        if len(plan.branches) != 1:
            excluded["multiple_gold_query_branches"] += 1
            continue
        if (
            intervention["intervention_type"] != "gold_query_pattern_ablation"
            or len(intervention["target_conditions"]) != 1
        ):
            excluded["nonstandard_k1_intervention"] += 1
            continue

        graph = [normalize_triple(triple) for triple in dataset_row["graph"]]
        graph_gold = graph_conditioned_answers(dataset_row["answer"], graph)
        mid_bindings = {
            str(mid).strip(): str(node).strip()
            for mid, node in witness_rows[example_id]["mid_bindings"].items()
        }

        def witness_exists(candidate_graph: list[Triple]) -> bool:
            result = evaluate_query(
                plan,
                [list(triple) for triple in candidate_graph],
                mid_bindings,
                graph_gold,
                reverse_properties,
            )
            return bool(result["has_complete_witness"])

        sequence = select_progressive_sequence(
            graph,
            plan=plan,
            reverse_properties=reverse_properties,
            protected_nodes=graph_gold,
            witness_exists=witness_exists,
            first_condition=intervention["target_conditions"][0],
            first_critical_indices=intervention["critical_graph_indices"],
        )
        if sequence is None:
            excluded["no_strict_second_required_condition"] += 1
            continue

        k1 = sequence["levels"][1]
        k2 = sequence["levels"][2]
        critical_indices = k2["critical_graph_indices"]
        critical_graph = _without_indices(graph, set(critical_indices))
        level_manifest = {
            "id": example_id,
            "question": dataset_row.get("question"),
            "compositionality_type": intervention["compositionality_type"],
            "graph_conditioned_gold": graph_gold,
            "system_top1": intervention["system_top1"],
            "intervention_type": "progressive_gold_query_pattern_ablation",
            "severity": 2,
            "target_conditions": sequence["conditions"],
            "k1_critical_graph_indices": k1["critical_graph_indices"],
            "k1_critical_triples": [
                list(graph[index - 1]) for index in k1["critical_graph_indices"]
            ],
            "critical_graph_indices": critical_indices,
            "critical_triples": [list(graph[index - 1]) for index in critical_indices],
            "added_graph_indices": k2["added_graph_indices"],
            "added_triples": [list(graph[index - 1]) for index in k2["added_graph_indices"]],
            "original_graph_triple_count": len(graph),
            "critical_removed_count": len(critical_indices),
            "critical_removed_fraction": len(critical_indices) / len(graph),
            "validations": {
                "gplus_has_complete_witness": True,
                "k1_has_complete_witness": False,
                "k2_has_complete_witness": False,
                "gold_answer_nodes_preserved": k2["gold_answer_nodes_preserved"],
            },
        }
        cohort_rows.append(
            {
                "id": example_id,
                "question": dataset_row.get("question"),
                "compositionality_type": intervention["compositionality_type"],
                "graph_conditioned_gold": graph_gold,
                "system_top1": intervention["system_top1"],
                "conditions": sequence["conditions"],
                "original_graph_triple_count": len(graph),
                "levels": {
                    "0": {"critical_removed_count": 0},
                    "1": {
                        "critical_graph_indices": k1["critical_graph_indices"],
                        "critical_removed_count": len(k1["critical_graph_indices"]),
                    },
                    "2": {
                        "critical_graph_indices": critical_indices,
                        "added_graph_indices": k2["added_graph_indices"],
                        "critical_removed_count": len(critical_indices),
                        "critical_removed_fraction": len(critical_indices) / len(graph),
                        "gold_answer_nodes_preserved": k2["gold_answer_nodes_preserved"],
                    },
                },
            }
        )
        k2_gminus_rows.append(
            {**dataset_row, "graph": [list(triple) for triple in critical_graph]}
        )
        k2_interventions.append(level_manifest)

    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "cohort.jsonl", cohort_rows)
    (output_dir / "k2").mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "k2" / "gminus.jsonl", k2_gminus_rows)
    write_jsonl(output_dir / "k2" / "interventions.jsonl", k2_interventions)

    k2_counts = [row["levels"]["2"]["critical_removed_count"] for row in cohort_rows]
    k2_fractions = [
        row["levels"]["2"]["critical_removed_fraction"] for row in cohort_rows
    ]
    answer_retained = sum(
        row["levels"]["2"]["gold_answer_nodes_preserved"] for row in cohort_rows
    )
    summary = {
        "protocol": {
            "name": "cwq-rq2-progressive-severity-v2",
            "source_population": "shared RQ2 cohort with correct predictions and sufficient returned evidence",
            "severity_levels": [0, 1, 2],
            "reused_levels": [0, 1],
            "new_inference_level": 2,
            "critical_graphs_nested": True,
            "sham_required": False,
            "maximum_removed_fraction": None,
            "answer_node_retention_is_diagnostic": True,
            "sequence_order": (
                "prefer answer-node retention, then fewer cumulative deleted edges, "
                "then added pattern index"
            ),
            "model_outputs_inspected_during_selection": False,
        },
        "source_rows": len(gplus_rows),
        "retained_rows": len(cohort_rows),
        "excluded": dict(sorted(excluded.items())),
        "k2_diagnostics": {
            "answer_nodes_preserved": answer_retained,
            "answer_nodes_not_preserved": len(cohort_rows) - answer_retained,
            "median_removed_edges": median(k2_counts) if k2_counts else None,
            "median_removed_percent": 100 * median(k2_fractions) if k2_fractions else None,
            "at_most_5_percent": sum(value <= 0.05 for value in k2_fractions),
            "above_5_percent": sum(value > 0.05 for value in k2_fractions),
        },
        "files": {
            "cohort": {"path": "cohort.jsonl", "rows": len(cohort_rows)},
            "k2_gminus": {"path": "k2/gminus.jsonl", "rows": len(k2_gminus_rows)},
            "k2_interventions": {
                "path": "k2/interventions.jsonl",
                "rows": len(k2_interventions),
            },
        },
        "sources": {
            "gplus": str(gplus_path.resolve()),
            "missing_interventions": str(missing_interventions_path.resolve()),
            "original_cwq": str(original_cwq_path.resolve()),
            "witness_details": str(witness_details_path.resolve()),
            "reverse_properties": str(reverse_properties_path.resolve()),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gplus", required=True, type=Path)
    parser.add_argument("--missing-interventions", required=True, type=Path)
    parser.add_argument("--original-cwq", required=True, type=Path)
    parser.add_argument("--witness-details", required=True, type=Path)
    parser.add_argument("--reverse-properties", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = build_progressive_bundle(
        gplus_path=args.gplus,
        missing_interventions_path=args.missing_interventions,
        original_cwq_path=args.original_cwq,
        witness_details_path=args.witness_details,
        reverse_properties_path=args.reverse_properties,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
