"""Construct CWQ graph pairs that break only shared-answer bindings."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Iterable

from dataset_split.cwq import (
    Comparison,
    Pattern,
    QueryPlan,
    candidate_matches,
    canonical_text,
    comparison_passes,
    evaluate_query,
    iter_witnesses,
    load_original_cwq,
    load_reverse_properties,
    parse_query,
)
from src.io import index_rows, read_jsonl, write_jsonl
from .cwq_witness_breaking import (
    Triple,
    _answer_nodes_present,
    _nearest_rank,
    _without_indices,
    select_non_query_sham,
)
from eva.evidence import (
    graph_conditioned_answers,
    normalize_triple,
)


MAX_SEARCH_STATES = 100_000
SHARED_BACKGROUND_RELATIONS = {"common.topic.notable_types"}


class BindingConflictSearchLimit(RuntimeError):
    pass


def _variables(pattern: Pattern) -> set[str]:
    return {
        term
        for term in (pattern.subject, pattern.object)
        if term.startswith("?")
    }


def split_binding_branches(
    patterns: tuple[Pattern, ...], answer_variable: str
) -> tuple[tuple[Pattern, ...], ...]:
    """Split a conjunctive query at its shared answer-role variable.

    Patterns connected through another query variable stay in the same branch.
    Constants do not merge branches: they constrain a branch but do not carry a
    runtime binding between two requirements.
    """

    foreground = tuple(
        pattern
        for pattern in patterns
        if pattern.relation not in SHARED_BACKGROUND_RELATIONS
    )
    if not foreground:
        raise ValueError("a binding query must contain at least one pattern")

    remaining = set(range(len(foreground)))
    branches: list[tuple[Pattern, ...]] = []
    while remaining:
        component = {min(remaining)}
        component_variables = _variables(foreground[next(iter(component))]) - {
            answer_variable
        }
        changed = True
        while changed:
            changed = False
            for index in sorted(remaining - component):
                variables = _variables(foreground[index]) - {answer_variable}
                if component_variables & variables:
                    component.add(index)
                    component_variables.update(variables)
                    changed = True
        branch = tuple(foreground[index] for index in sorted(component))
        if not any(
            answer_variable in (pattern.subject, pattern.object)
            for pattern in branch
        ):
            raise ValueError("query contains a component disconnected from the answer variable")
        branches.append(branch)
        remaining -= component

    return tuple(sorted(branches, key=lambda branch: branch[0].index))


def _comparison_variables(comparison: Comparison) -> set[str]:
    return {
        operand.term
        for operand in (comparison.left, comparison.right)
        if operand.term.startswith("?")
    }


def _branch_witnesses(
    graph: list[Triple],
    *,
    branch: tuple[Pattern, ...],
    plan: QueryPlan,
    mid_bindings: dict[str, str],
    reverse_properties: dict[str, str],
) -> dict[str, list[set[int]]]:
    graph_rows = [list(triple) for triple in graph]
    background = tuple(
        pattern
        for pattern in plan.branches[0]
        if pattern.relation in SHARED_BACKGROUND_RELATIONS
    )
    evaluation_branch = tuple(sorted((*branch, *background), key=lambda item: item.index))
    foreground_indices = {pattern.index for pattern in branch}
    matches = candidate_matches(evaluation_branch, graph_rows, reverse_properties)
    branch_variables = set().union(
        *(_variables(pattern) for pattern in evaluation_branch)
    )
    applicable_comparisons = [
        comparison
        for comparison in plan.comparisons
        if _comparison_variables(comparison) <= branch_variables
    ]
    witnesses: dict[str, list[set[int]]] = {}
    for chosen, bindings in iter_witnesses(
        evaluation_branch,
        matches,
        graph_rows,
        mid_bindings,
        None,
        plan.answer_variable,
    ):
        comparison_bindings = {
            **{f"ns:{mid}": node for mid, node in mid_bindings.items()},
            **bindings,
        }
        if any(
            comparison_passes(comparison, comparison_bindings) is not True
            for comparison in applicable_comparisons
        ):
            continue
        answer = bindings.get(plan.answer_variable)
        if answer is None:
            raise ValueError("binding branch does not bind the answer variable")
        witnesses.setdefault(canonical_text(answer), []).append(
            {
                match.graph_index
                for match in chosen
                if foreground_indices & set(match.covered)
            }
        )
    return witnesses


def binding_sets(
    graph: list[Triple],
    *,
    branches: tuple[tuple[Pattern, ...], ...],
    plan: QueryPlan,
    mid_bindings: dict[str, str],
    reverse_properties: dict[str, str],
) -> list[set[str]]:
    return [
        set(
            _branch_witnesses(
                graph,
                branch=branch,
                plan=plan,
                mid_bindings=mid_bindings,
                reverse_properties=reverse_properties,
            )
        )
        for branch in branches
    ]


def _joint_bindings(branch_sets: Iterable[set[str]]) -> set[str]:
    sets = list(branch_sets)
    if not sets:
        return set()
    return set.intersection(*sets)


def select_binding_conflict_pair(
    graph: list[Triple],
    *,
    plan: QueryPlan,
    mid_bindings: dict[str, str],
    reverse_properties: dict[str, str],
    protected_nodes: list[str],
) -> dict | None:
    """Find an equal-size critical/sham pair for a conjunctive gold query."""

    if len(plan.branches) != 1 or plan.order is not None or plan.optional_exists:
        return None
    branches = split_binding_branches(plan.branches[0], plan.answer_variable)
    if len(branches) < 2:
        return None
    branch_variables = [
        set().union(*(_variables(pattern) for pattern in branch))
        for branch in branches
    ]
    if any(
        variables and not any(variables <= available for available in branch_variables)
        for variables in map(_comparison_variables, plan.comparisons)
    ):
        return None

    protected_keys = {canonical_text(node) for node in protected_nodes}

    def full_replay(candidate_graph: list[Triple]) -> bool:
        return bool(
            evaluate_query(
                plan,
                [list(triple) for triple in candidate_graph],
                mid_bindings,
                protected_nodes,
                reverse_properties,
            )["has_complete_witness"]
        )

    def state(
        removed: frozenset[int],
    ) -> tuple[list[set[str]], list[dict[str, list[set[int]]]], list[Triple]]:
        active = [index for index in range(1, len(graph) + 1) if index not in removed]
        candidate_graph = [graph[index - 1] for index in active]
        witness_groups = [
            _branch_witnesses(
                candidate_graph,
                branch=branch,
                plan=plan,
                mid_bindings=mid_bindings,
                reverse_properties=reverse_properties,
            )
            for branch in branches
        ]
        for group in witness_groups:
            for witnesses in group.values():
                for witness in witnesses:
                    local = set(witness)
                    witness.clear()
                    witness.update(active[index - 1] for index in local)
        return [set(group) for group in witness_groups], witness_groups, candidate_graph

    gplus_sets, _gplus_witnesses, _gplus_graph = state(frozenset())
    gplus_joint = _joint_bindings(gplus_sets)
    if (
        any(not branch_set for branch_set in gplus_sets)
        or not (gplus_joint & protected_keys)
        or not full_replay(graph)
    ):
        return None

    queue = deque([frozenset()])
    visited = {frozenset()}
    explored = 0
    chosen: tuple[frozenset[int], list[set[str]], list[Triple]] | None = None
    while queue:
        removed = queue.popleft()
        explored += 1
        if explored > MAX_SEARCH_STATES:
            raise BindingConflictSearchLimit(
                f"binding-conflict search exceeded {MAX_SEARCH_STATES} states"
            )
        branch_sets, witness_groups, candidate_graph = state(removed)
        if any(not branch_set for branch_set in branch_sets):
            continue
        if not _answer_nodes_present(candidate_graph, protected_nodes):
            continue
        joint = _joint_bindings(branch_sets)
        if removed and not joint:
            if not full_replay(candidate_graph):
                chosen = removed, branch_sets, candidate_graph
                break
            continue

        target = min(joint)
        next_indices = sorted(
            {
                index
                for group in witness_groups
                for witness in group[target]
                for index in witness
            }
        )
        for index in next_indices:
            successor = removed | {index}
            if successor not in visited:
                visited.add(successor)
                queue.append(successor)

    if chosen is None:
        return None

    critical_indices = sorted(chosen[0])
    critical_sets = chosen[1]
    critical_graph = chosen[2]
    sham = select_non_query_sham(
        graph,
        plan=plan,
        reverse_properties=reverse_properties,
        removal_count=len(critical_indices),
        protected_nodes=protected_nodes,
        witness_exists=full_replay,
    )
    sham_indices = [int(index) for index in sham["sham_graph_indices"]]
    if not sham_indices:
        return None
    sham_graph = _without_indices(graph, set(sham_indices))
    sham_sets = binding_sets(
        sham_graph,
        branches=branches,
        plan=plan,
        mid_bindings=mid_bindings,
        reverse_properties=reverse_properties,
    )
    sham_joint = _joint_bindings(sham_sets)
    if (
        any(not branch_set for branch_set in sham_sets)
        or not (sham_joint & protected_keys)
        or not _answer_nodes_present(sham_graph, protected_nodes)
        or not full_replay(sham_graph)
    ):
        return None

    return {
        "binding_branches": [
            [
                {
                    "pattern_index": pattern.index,
                    "subject": pattern.subject,
                    "relation": pattern.relation,
                    "object": pattern.object,
                }
                for pattern in branch
            ]
            for branch in branches
        ],
        "shared_background_patterns": [
            {
                "pattern_index": pattern.index,
                "subject": pattern.subject,
                "relation": pattern.relation,
                "object": pattern.object,
            }
            for pattern in plan.branches[0]
            if pattern.relation in SHARED_BACKGROUND_RELATIONS
        ],
        "critical_graph_indices": critical_indices,
        "critical_triples": [graph[index - 1] for index in critical_indices],
        **sham,
        "search_states_explored": explored,
        "validations": {
            "gplus_branch_binding_counts": [len(values) for values in gplus_sets],
            "gplus_joint_bindings": sorted(gplus_joint),
            "gconflict_branch_binding_counts": [len(values) for values in critical_sets],
            "gconflict_joint_bindings": [],
            "gconflict_has_complete_witness": False,
            "gsham_branch_binding_counts": [len(values) for values in sham_sets],
            "gsham_joint_bindings": sorted(sham_joint),
            "gsham_has_complete_witness": True,
            "gold_answer_nodes_preserved": _answer_nodes_present(
                critical_graph, protected_nodes
            ),
        },
    }


def _load_plan_ids(plan_dir: Path) -> tuple[set[str], set[str]]:
    result: set[str] = set()
    seen: set[str] = set()
    for path in sorted(plan_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        example_id = str(row["id"])
        if example_id in seen:
            raise ValueError(f"duplicate Ours plan id: {example_id}")
        seen.add(example_id)
        steps = row.get("plan", {}).get("steps", [])
        if any(str(step.get("op", "")).casefold() == "intersect" for step in steps):
            result.add(example_id)
    return result, seen


def _unsupported_reason(plan: QueryPlan) -> str | None:
    if len(plan.branches) != 1:
        return "union_query"
    if plan.order is not None:
        return "ordered_query"
    if plan.optional_exists:
        return "optional_exists_query"
    try:
        branches = split_binding_branches(plan.branches[0], plan.answer_variable)
    except ValueError:
        return "answer_disconnected_query_component"
    if len(branches) < 2:
        return "fewer_than_two_contract_branches"
    branch_variables = [
        set().union(*(_variables(pattern) for pattern in branch))
        for branch in branches
    ]
    if any(
        variables and not any(variables <= available for available in branch_variables)
        for variables in map(_comparison_variables, plan.comparisons)
    ):
        return "cross_branch_comparison"
    return None


def build_binding_conflicts(
    *,
    k_path: Path,
    common_pool_path: Path,
    original_cwq_path: Path,
    witness_details_path: Path,
    reverse_properties_path: Path,
    ours_plan_dir: Path,
    output_dir: Path,
) -> dict:
    """Build and freeze every valid conjunction-row Binding Conflict pair."""

    target_names = {
        "eligibility.jsonl",
        "interventions.jsonl",
        "gplus.jsonl",
        "gbinding.jsonl",
        "gsham.jsonl",
        "summary.json",
    }
    if output_dir.exists() and any((output_dir / name).exists() for name in target_names):
        raise FileExistsError(f"refusing to overwrite Binding Conflict artifacts: {output_dir}")

    common_rows = list(read_jsonl(common_pool_path))
    common_by_id = index_rows(common_rows, "id", "common-pool")
    if any(not row.get("common_certified_hit") for row in common_rows):
        raise ValueError("common pool contains a row outside the correct-and-supported RQ2 cohort")
    original_by_id = load_original_cwq(original_cwq_path)
    witness_by_id = index_rows(read_jsonl(witness_details_path), "id", "witness")
    reverse_properties = load_reverse_properties(reverse_properties_path)
    explicit_intersect_ids, plan_ids = _load_plan_ids(ours_plan_dir)
    missing_plans = set(common_by_id) - plan_ids
    if missing_plans:
        raise ValueError(f"source-pool ids absent from Ours plans: {sorted(missing_plans)[:3]}")

    eligibility: list[dict] = []
    manifests: list[dict] = []
    gplus_rows: list[dict] = []
    gbinding_rows: list[dict] = []
    gsham_rows: list[dict] = []
    exclusions = Counter()
    conjunction_ids = {
        example_id
        for example_id in common_by_id
        if str(original_by_id[example_id].get("compositionality_type") or "").casefold()
        == "conjunction"
    }

    for dataset_row in read_jsonl(k_path):
        example_id = str(dataset_row["id"])
        if example_id not in conjunction_ids:
            continue
        pool_row = common_by_id[example_id]
        original = original_by_id[example_id]
        explicit_intersect = example_id in explicit_intersect_ids
        base_eligibility = {
            "id": example_id,
            "question": dataset_row.get("question"),
            "ours_plan_has_explicit_intersect": explicit_intersect,
        }
        plan = parse_query(original["sparql"])
        unsupported = _unsupported_reason(plan)
        if unsupported is not None:
            exclusions[unsupported] += 1
            eligibility.append(
                {**base_eligibility, "status": "excluded", "reason": unsupported}
            )
            continue

        graph = [normalize_triple(triple) for triple in dataset_row["graph"]]
        graph_gold = graph_conditioned_answers(dataset_row["answer"], graph)
        witness = witness_by_id[example_id]
        mid_bindings = {
            str(mid).strip(): str(node).strip()
            for mid, node in witness["mid_bindings"].items()
        }
        try:
            pair = select_binding_conflict_pair(
                graph,
                plan=plan,
                mid_bindings=mid_bindings,
                reverse_properties=reverse_properties,
                protected_nodes=graph_gold,
            )
        except BindingConflictSearchLimit:
            reason = "search_limit"
            exclusions[reason] += 1
            eligibility.append(
                {**base_eligibility, "status": "excluded", "reason": reason}
            )
            continue
        except ValueError as error:
            reason = "replay_unresolved"
            exclusions[reason] += 1
            eligibility.append(
                {
                    **base_eligibility,
                    "status": "excluded",
                    "reason": reason,
                    "detail": str(error),
                }
            )
            continue
        if pair is None:
            reason = "no_validated_critical_sham_pair"
            exclusions[reason] += 1
            eligibility.append(
                {**base_eligibility, "status": "excluded", "reason": reason}
            )
            continue

        critical_indices = [int(index) for index in pair["critical_graph_indices"]]
        sham_indices = [int(index) for index in pair["sham_graph_indices"]]
        conflict_graph = _without_indices(graph, set(critical_indices))
        sham_graph = _without_indices(graph, set(sham_indices))
        eligibility.append(
            {
                **base_eligibility,
                "status": "validated",
                "reason": None,
                "critical_removed_count": len(critical_indices),
            }
        )
        manifests.append(
            {
                "id": example_id,
                "question": dataset_row.get("question"),
                "compositionality_type": "conjunction",
                "ours_plan_has_explicit_intersect": explicit_intersect,
                "graph_conditioned_gold": graph_gold,
                "system_top1": {"ours": pool_row["systems"]["ours"]["top1"]},
                "intervention_type": "binding_conflict",
                "binding_branches": pair["binding_branches"],
                "shared_background_patterns": pair["shared_background_patterns"],
                "critical_graph_indices": critical_indices,
                "critical_triples": [list(triple) for triple in pair["critical_triples"]],
                "sham_match": pair["sham_match"],
                "sham_graph_indices": sham_indices,
                "sham_triples": [list(triple) for triple in pair["sham_triples"]],
                "original_graph_triple_count": len(graph),
                "critical_removed_count": len(critical_indices),
                "critical_removed_fraction": len(critical_indices) / len(graph),
                "search_states_explored": pair["search_states_explored"],
                "validations": pair["validations"],
            }
        )
        gplus_rows.append(dataset_row)
        gbinding_rows.append(
            {**dataset_row, "graph": [list(triple) for triple in conflict_graph]}
        )
        gsham_rows.append({**dataset_row, "graph": [list(triple) for triple in sham_graph]})

    if len(eligibility) != len(conjunction_ids):
        found = {row["id"] for row in eligibility}
        missing = sorted(conjunction_ids - found)
        raise ValueError(f"conjunction ids absent from CWQ-K input: {missing[:3]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "eligibility.jsonl", eligibility)
    write_jsonl(output_dir / "interventions.jsonl", manifests)
    write_jsonl(output_dir / "gplus.jsonl", gplus_rows)
    write_jsonl(output_dir / "gbinding.jsonl", gbinding_rows)
    write_jsonl(output_dir / "gsham.jsonl", gsham_rows)

    removal_counts = [float(row["critical_removed_count"]) for row in manifests]
    removal_fractions = [
        100.0 * float(row["critical_removed_fraction"]) for row in manifests
    ]
    summary = {
        "protocol": {
            "name": "cwq-v3-binding-conflict-intervention-v1",
            "source_pool": "provided frozen intervention cohort",
            "critical_intervention": (
                "preserve a non-empty answer-role binding set for every gold-contract "
                "branch while making their joint intersection empty"
            ),
            "shared_background": (
                "common.topic.notable_types constrains every branch and is never a "
                "critical-edit target"
            ),
            "sham_intervention": (
                "remove the same number of non-query edges while preserving a gold "
                "joint binding and successful full replay"
            ),
        },
        "source_common_certified_rows": len(common_rows),
        "conjunction_candidates": len(conjunction_ids),
        "ours_explicit_intersect_candidates": len(conjunction_ids & explicit_intersect_ids),
        "validated_interventions": len(manifests),
        "validated_from_explicit_intersect_candidates": sum(
            int(row["ours_plan_has_explicit_intersect"]) for row in manifests
        ),
        "excluded": dict(sorted(exclusions.items())),
        "by_binding_branch_count": dict(
            sorted(Counter(len(row["binding_branches"]) for row in manifests).items())
        ),
        "by_sham_match": dict(
            sorted(Counter(row["sham_match"] for row in manifests).items())
        ),
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
        "sources": {
            "cwq_k": str(k_path.resolve()),
            "common_pool": str(common_pool_path.resolve()),
            "original_cwq": str(original_cwq_path.resolve()),
            "witness_details": str(witness_details_path.resolve()),
            "reverse_properties": str(reverse_properties_path.resolve()),
            "ours_plan_dir": str(ours_plan_dir.resolve()),
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
    parser.add_argument("--ours-plan-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_binding_conflicts(
        k_path=args.k,
        common_pool_path=args.common_pool,
        original_cwq_path=args.original_cwq,
        witness_details_path=args.witness_details,
        reverse_properties_path=args.reverse_properties,
        ours_plan_dir=args.ours_plan_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
