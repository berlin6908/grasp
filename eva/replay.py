"""Replay benchmark constraints over Grasp's cited evidence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from dataset_split.cwq import evaluate_query, parse_query, validated_string_list
from dataset_split.webqsp import build_mid_bindings, validate_supported_sparql
from dataset_split.two_wiki import normalize_official_evidence
from .evidence import Triple, semantic_evidence


def _replay(
    *,
    plan,
    triples: list[Triple],
    mid_bindings: dict[str, str],
    gold_answers: list[str],
    reverse_properties: dict[str, str],
) -> str:
    try:
        result = evaluate_query(
            plan,
            [list(triple) for triple in triples],
            mid_bindings,
            gold_answers,
            reverse_properties,
        )
    except ValueError:
        return "unresolved"
    except Exception:
        return "error"
    return "success" if result["has_complete_witness"] else "no_witness"


def replay_cwq(
    dataset_row: dict,
    evidence: list[tuple[str, str, str]],
    target_answers: list[str],
    graph: list[tuple[str, str, str]],
    *,
    original_by_id: dict[str, dict],
    witness_by_id: dict[str, dict],
    reverse_properties: dict[str, str],
    relation_matches: dict[str, frozenset[str]] | None = None,
) -> str:
    """Replay the existing CWQ gold-SPARQL contract."""

    if not evidence or not target_answers:
        return "no_evidence"
    example_id = str(dataset_row["id"])
    try:
        result = evaluate_query(
            parse_query(original_by_id[example_id]["sparql"]),
            [list(triple) for triple in semantic_evidence(evidence, graph)],
            {
                str(mid).strip(): str(node).strip()
                for mid, node in witness_by_id[example_id]["mid_bindings"].items()
            },
            target_answers,
            reverse_properties,
            relation_matches,
        )
    except ValueError:
        return "unresolved"
    except Exception:
        return "error"
    return "success" if result["has_complete_witness"] else "no_witness"


def _complete_parse_records(question: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for parse_record in question["Parses"]:
        if (parse_record.get("AnnotatorComment") or {}).get(
            "ParseQuality"
        ) == "Complete":
            yield parse_record


def replay_webqsp(
    dataset_row: dict,
    evidence: list[tuple[str, str, str]],
    target_answers: list[str],
    graph: list[tuple[str, str, str]],
    *,
    official_questions: dict[str, dict],
    supplemental_mid_names: dict[str, str],
    reverse_properties: dict[str, str],
) -> str:
    """Run the existing replay core for every official Complete parse."""

    if not evidence or not target_answers:
        return "no_evidence"
    example_id = str(dataset_row["id"])
    try:
        q_entities = validated_string_list(
            dataset_row.get("q_entity"),
            field="q_entity",
            example_id=example_id,
        )
        parses = list(_complete_parse_records(official_questions[example_id]))
        replay_graph = semantic_evidence(evidence, graph)
    except (KeyError, TypeError, ValueError):
        return "unresolved"

    statuses = []
    for parse_record in parses:
        try:
            plan = parse_query(
                validate_supported_sparql(parse_record.get("Sparql"))
            )
            mid_bindings = build_mid_bindings(
                parse_record, q_entities, supplemental_mid_names
            )
        except (TypeError, ValueError):
            statuses.append("unresolved")
            continue
        status = _replay(
            plan=plan,
            triples=replay_graph,
            mid_bindings=mid_bindings,
            gold_answers=target_answers,
            reverse_properties=reverse_properties,
        )
        if status == "success":
            return status
        statuses.append(status)
    if "no_witness" in statuses:
        return "no_witness"
    if "error" in statuses:
        return "error"
    return "unresolved"


def _covers_required_triples(
    label: dict,
    triples: list[tuple[str, str, str]],
) -> bool:
    reported = {
        tuple(normalize_official_evidence(value) for value in triple)
        for triple in triples
    }
    facts = label.get("fact_results") or []
    return bool(facts) and all(
        not reported.isdisjoint(
            tuple(triple) for triple in fact["matched_graph_triples"]
        )
        for fact in facts
    )


def replay_two_wiki(
    dataset_row: dict,
    evidence: list[tuple[str, str, str]],
    target_answers: list[str],
    graph: list[tuple[str, str, str]],
    *,
    labels: dict[str, dict],
) -> str:
    """Use the same required-triple coverage as the recorded 2Wiki evaluator."""

    complete = _covers_required_triples(labels[str(dataset_row["id"])], evidence)
    return "success" if complete else "no_witness"
