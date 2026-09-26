"""2WikiMultiHopQA S/I/X split under the paper's frozen required-evidence rule.

The matcher uses official endpoint identities/aliases, normalized exact relations
and direction, and the closed-world v08 policy. Historical label fields are kept.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from src.io import index_rows, load_jsonl, write_json, write_jsonl

VERSION = "2wiki_required_evidence_closed_world_v08"
_QID_PATTERN = re.compile(r"Q\d+")


def normalize_official_evidence(value: object) -> str:
    """Reproduce 2Wiki evaluator v1.1 ``normalize_evi`` for one string."""
    lowered = str(value).lower()
    without_ascii_punctuation = "".join(
        character
        for character in lowered
        if character not in string.punctuation
    )
    return " ".join(without_ascii_punctuation.split())


def build_entity_alias_index(
    alias_rows: Iterable[dict],
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """Index the aliases and demonyms used by 2Wiki evaluator v1.1."""
    aliases_by_id: dict[str, set[str]] = defaultdict(set)
    ids_by_alias: dict[str, set[str]] = defaultdict(set)
    for row in alias_rows:
        qid = row["Q_id"]
        for value in (*row["aliases"], *row["demonyms"]):
            surface = normalize_official_evidence(value)
            if surface:
                aliases_by_id[qid].add(surface)
                ids_by_alias[surface].add(qid)
    return (
        {
            qid: frozenset(surfaces)
            for qid, surfaces in aliases_by_id.items()
        },
        {
            surface: frozenset(qids)
            for surface, qids in ids_by_alias.items()
        },
    )


def _qid_or_none(value: object) -> str | None:
    candidate = str(value).strip()
    return candidate if _QID_PATTERN.fullmatch(candidate) else None


def _row_candidate_qids(official_row: dict) -> frozenset[str]:
    candidates: set[str] = set()
    for value in str(official_row.get("entity_ids") or "").split("_"):
        if qid := _qid_or_none(value):
            candidates.add(qid)
    if qid := _qid_or_none(official_row.get("answer_id")):
        candidates.add(qid)
    for evidence_id in official_row.get("evidences_id") or []:
        for value in (evidence_id[0], evidence_id[2]):
            if qid := _qid_or_none(value):
                candidates.add(qid)
    return frozenset(candidates)


def _resolve_endpoint_qid(
    surface: object,
    explicit_qid: str | None,
    row_candidate_qids: frozenset[str],
    aliases_by_id: dict[str, frozenset[str]],
) -> tuple[str | None, str, list[str]]:
    normalized_surface = normalize_official_evidence(surface)
    alias_candidates = sorted(
        qid
        for qid in row_candidate_qids
        if normalized_surface in aliases_by_id.get(qid, frozenset())
    )
    if explicit_qid is not None:
        resolved = explicit_qid
        basis = (
            "official_evidence_id_supported_by_alias"
            if explicit_qid in alias_candidates
            else "official_evidence_id"
        )
    elif len(alias_candidates) == 1:
        resolved = alias_candidates[0]
        basis = "row_unique_official_alias"
    else:
        resolved = None
        basis = "ambiguous_row_alias" if alias_candidates else "unresolved"
    return resolved, basis, alias_candidates


def _matched_graph_nodes(
    gold_surface: object,
    qid: str | None,
    graph_nodes: set[str],
    row_candidate_qids: frozenset[str],
    aliases_by_id: dict[str, frozenset[str]],
    ids_by_alias: dict[str, frozenset[str]],
    *,
    qid_is_explicit: bool,
) -> set[str]:
    direct_surface = normalize_official_evidence(gold_surface)
    matches: set[str] = set()
    direct_row_alias_qids = {
        candidate_qid
        for candidate_qid in row_candidate_qids
        if direct_surface in aliases_by_id.get(candidate_qid, frozenset())
    }
    direct_is_unique = qid_is_explicit or (
        len(direct_row_alias_qids) <= 1
        if qid is None
        else direct_row_alias_qids in (set(), {qid})
    )
    if direct_surface in graph_nodes and direct_is_unique:
        matches.add(direct_surface)
    if qid is None:
        return matches

    for surface in aliases_by_id.get(qid, frozenset()).intersection(graph_nodes):
        if surface == direct_surface:
            continue
        if (
            qid_is_explicit
            or ids_by_alias.get(surface, frozenset()) == frozenset((qid,))
        ):
            matches.add(surface)
    return matches


def classify_required_evidence(
    official_row: dict,
    dog_row: dict,
    aliases_by_id: dict[str, frozenset[str]],
    ids_by_alias: dict[str, frozenset[str]],
) -> dict:
    """Classify one row with deterministic official entity equivalence."""
    evidences = official_row["evidences"]
    evidence_ids = official_row.get("evidences_id") or []
    if not evidences:
        raise ValueError("2Wiki row has no official evidence triples")
    if evidence_ids and len(evidence_ids) != len(evidences):
        raise ValueError("evidences_id does not align with evidences")
    if evidence_ids:
        for fact_index, (evidence, evidence_id) in enumerate(
            zip(evidences, evidence_ids, strict=True)
        ):
            if len(evidence_id) != 3:
                raise ValueError(
                    f"evidences_id fact {fact_index} is not a triple"
                )
            if evidence_id[1] != evidence[1]:
                raise ValueError(
                    f"evidences_id relation does not match evidences at fact "
                    f"{fact_index}"
                )

    graph_triples = [
        (
            normalize_official_evidence(subject),
            normalize_official_evidence(relation),
            normalize_official_evidence(obj),
        )
        for subject, relation, obj in dog_row["graph"]
    ]
    graph_nodes = {
        endpoint
        for subject, _relation, obj in graph_triples
        for endpoint in (subject, obj)
    }
    row_candidate_qids = _row_candidate_qids(official_row)

    fact_results: list[dict] = []
    row_official_surface_id_disagreements: list[dict] = []
    for fact_index, (subject, relation, obj) in enumerate(evidences):
        evidence_id = evidence_ids[fact_index] if evidence_ids else (None, None, None)
        explicit_subject_qid = _qid_or_none(evidence_id[0])
        explicit_object_qid = _qid_or_none(evidence_id[2])
        subject_qid, subject_qid_basis, subject_qid_candidates = (
            _resolve_endpoint_qid(
                subject,
                explicit_subject_qid,
                row_candidate_qids,
                aliases_by_id,
            )
        )
        object_qid, object_qid_basis, object_qid_candidates = (
            _resolve_endpoint_qid(
                obj,
                explicit_object_qid,
                row_candidate_qids,
                aliases_by_id,
            )
        )
        official_surface_id_disagreements: list[str] = []
        if (
            explicit_subject_qid is not None
            and subject_qid_candidates
            and explicit_subject_qid not in subject_qid_candidates
        ):
            official_surface_id_disagreements.append("subject")
        if (
            explicit_object_qid is not None
            and object_qid_candidates
            and explicit_object_qid not in object_qid_candidates
        ):
            official_surface_id_disagreements.append("object")
        if official_surface_id_disagreements:
            row_official_surface_id_disagreements.append({
                "fact_index": fact_index,
                "endpoints": official_surface_id_disagreements,
            })

        subject_matches = _matched_graph_nodes(
            subject,
            subject_qid,
            graph_nodes,
            row_candidate_qids,
            aliases_by_id,
            ids_by_alias,
            qid_is_explicit=explicit_subject_qid is not None,
        )
        object_matches = _matched_graph_nodes(
            obj,
            object_qid,
            graph_nodes,
            row_candidate_qids,
            aliases_by_id,
            ids_by_alias,
            qid_is_explicit=explicit_object_qid is not None,
        )
        gold_relation = normalize_official_evidence(relation)
        matched_edges: dict[int, str] = {}
        competing_indices: list[int] = []
        unresolved_relation_indices: list[int] = []
        for edge_index, (graph_subject, graph_relation, graph_object) in enumerate(
            graph_triples
        ):
            ordered_pair = (
                graph_subject in subject_matches
                and graph_object in object_matches
            )
            reversed_pair = (
                graph_subject in object_matches
                and graph_object in subject_matches
            )
            if not ordered_pair and not reversed_pair:
                continue
            direction = "same" if ordered_pair else "reversed"
            competing_indices.append(edge_index)
            if direction == "same" and graph_relation == gold_relation:
                matched_edges[edge_index] = "exact_surface"
            else:
                unresolved_relation_indices.append(edge_index)

        missing_proof: str | None = None
        if official_surface_id_disagreements:
            status = "official_surface_id_conflict"
        elif matched_edges:
            status = "matched"
        elif subject_matches and object_matches:
            status = "missing"
            missing_proof = (
                "required_relation_or_direction_absent"
                if unresolved_relation_indices else "endpoint_pair_absent"
            )
        else:
            status = "missing"
            missing_proof = "endpoint_absent_under_official_equivalence"

        matched_indices = sorted(matched_edges)
        match_bases = [matched_edges[index] for index in matched_indices]
        fact_results.append({
            "fact_index": fact_index,
            "gold_triple": [str(subject), str(relation), str(obj)],
            "explicit_subject_qid": explicit_subject_qid,
            "explicit_object_qid": explicit_object_qid,
            "subject_qid": subject_qid,
            "object_qid": object_qid,
            "subject_qid_basis": subject_qid_basis,
            "object_qid_basis": object_qid_basis,
            "subject_qid_candidates": subject_qid_candidates,
            "object_qid_candidates": object_qid_candidates,
            "official_surface_id_disagreements": (
                official_surface_id_disagreements
            ),
            "subject_matches": sorted(subject_matches),
            "object_matches": sorted(object_matches),
            "gold_relation_pid": None,
            "status": status,
            "match_basis": match_bases[0] if match_bases else None,
            "match_bases": match_bases,
            "missing_proof": missing_proof,
            "matched_graph_edge_indices": matched_indices,
            "matched_graph_triples": [
                list(graph_triples[edge_index]) for edge_index in matched_indices
            ],
            "competing_graph_edge_indices": sorted(competing_indices),
            "resolved_relation_conflict_edge_indices": [],
            "unresolved_relation_edge_indices": sorted(
                unresolved_relation_indices
            ),
            "typed_value_conflict_edge_indices": [],
            "typed_value_unresolved_edge_indices": [],
        })

    facts_by_status = {
        status: [fact["fact_index"] for fact in fact_results if fact["status"] == status]
        for status in ("matched", "missing", "official_surface_id_conflict")
    }
    matched = facts_by_status["matched"]
    missing = facts_by_status["missing"]
    conflicts = facts_by_status["official_surface_id_conflict"]
    if conflicts:
        split, coverage = "X", "official_surface_id_conflict"
    elif missing:
        split, coverage = "U", "required_evidence_missing"
    else:
        split, coverage = "K", "complete_required_evidence"
    return {
        "split": split,
        "coverage_label": coverage,
        "evidence_count": len(evidences),
        "official_evidence_ids_available": bool(evidence_ids),
        "official_surface_id_disagreements": row_official_surface_id_disagreements,
        "matched_evidence_count": len(matched),
        "triple_coverage": len(matched) / len(evidences),
        "complete_required_evidence": len(matched) == len(evidences),
        "matched_fact_indices": matched,
        "missing_fact_indices": missing,
        "official_id_conflict_fact_indices": conflicts,
        "entity_unresolved_fact_indices": [],
        "relation_or_direction_unresolved_fact_indices": [],
        "value_alignment_unresolved_fact_indices": [],
        "unresolved_fact_indices": conflicts,
        "fact_results": fact_results,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dev", type=Path, required=True)
    parser.add_argument("--official-aliases", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    official = index_rows(
        json.loads(args.official_dev.read_text(encoding="utf-8")), "_id", "official 2Wiki",
    )
    aliases, ids = build_entity_alias_index(load_jsonl(args.official_aliases))
    dataset = index_rows(load_jsonl(args.dataset), "id", "2Wiki graph")
    labels = [
        {
            "index": index,
            "id": example_id,
            "type": official[example_id]["type"],
            **classify_required_evidence(official[example_id], row, aliases, ids),
        }
        for index, (example_id, row) in enumerate(dataset.items(), start=1)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / f"{VERSION}_labels.jsonl", labels)
    for split in ("K", "U", "X"):
        selected = [label for label in labels if label["split"] == split]
        write_jsonl(args.output_dir / f"{VERSION}_{split.lower()}_labels.jsonl", selected)
        write_jsonl(
            args.output_dir / f"{VERSION}_{split.lower()}.jsonl",
            (dataset[label["id"]] for label in selected),
        )
    summary = {
        "version": VERSION,
        "examples": len(labels),
        "split": dict(Counter(label["split"] for label in labels)),
    }
    write_json(args.output_dir / f"{VERSION}_summary.json", summary)
    print(json.dumps({
        "total": len(labels),
        **{label: summary["split"].get(stored, 0)
           for stored, label in (("K", "S"), ("U", "I"), ("X", "unresolved"))},
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
