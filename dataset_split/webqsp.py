#!/usr/bin/env python
"""Split folded RoG-WebQSP by replaying official gold SPARQL.

Each official complete semantic parse is evaluated independently on the
directed folded graph. A row belongs to S when any parse returns a
RoG gold answer with positive graph evidence for every answer-changing
condition. A row belongs to I only when every parse is evaluated and none
succeeds; parse, binding, or evaluation failures remain unresolved.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src import io
from .cwq import (
    EvaluationLimit,
    MID_RE,
    canonical_text,
    evaluate_query,
    load_reverse_properties,
    parse_query,
    validated_string_list,
)


MID = re.compile(r"[mg]\.[A-Za-z0-9_]+")


def load_webqsp_questions(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return io.index_rows(payload["Questions"], "QuestionId", "WebQSP")


def load_supplemental_mid_names(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {mid.strip(): name.strip() for mid, name in payload.items()}


def _topic_surface(parse_record: dict[str, Any], q_entities: list[str]) -> str:
    topic_name = parse_record.get("TopicEntityName")
    if isinstance(topic_name, str) and topic_name.strip():
        matches = [
            entity
            for entity in q_entities
            if canonical_text(entity) == canonical_text(topic_name)
        ]
        if matches:
            return matches[0]
    if len(q_entities) == 1:
        return q_entities[0]
    raise ValueError(
        "parse topic name does not identify one of the RoG q_entity values"
    )


def _add_binding(bindings: dict[str, str], mid: object, name: object) -> None:
    if not isinstance(mid, str) or not MID.fullmatch(mid):
        raise ValueError(f"invalid WebQSP MID: {mid!r}")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"WebQSP MID lacks a surface name: {mid}")
    surface = name.strip()
    previous = bindings.get(mid)
    if previous is not None and canonical_text(previous) != canonical_text(surface):
        raise ValueError(f"conflicting surface names for WebQSP MID {mid}")
    bindings[mid] = surface


def build_mid_bindings(
    parse_record: dict[str, Any],
    q_entities: list[str],
    supplemental_mid_names: dict[str, str],
) -> dict[str, str]:
    sparql = parse_record.get("Sparql")
    if not isinstance(sparql, str) or not sparql.strip():
        raise ValueError("WebQSP parse lacks SPARQL")
    query_mids = set(MID_RE.findall(sparql))
    bindings: dict[str, str] = {}

    topic_mid = parse_record.get("TopicEntityMid")
    if topic_mid in query_mids:
        _add_binding(bindings, topic_mid, _topic_surface(parse_record, q_entities))

    constraints = parse_record.get("Constraints")
    if not isinstance(constraints, list):
        raise ValueError("WebQSP parse Constraints is not a list")
    for constraint in constraints:
        if not isinstance(constraint, dict):
            raise ValueError("WebQSP constraint is not an object")
        if constraint.get("ArgumentType") != "Entity":
            continue
        mid = constraint.get("Argument")
        if mid in query_mids:
            _add_binding(bindings, mid, constraint.get("EntityName"))

    for mid in query_mids - bindings.keys():
        if mid in supplemental_mid_names:
            _add_binding(bindings, mid, supplemental_mid_names[mid])

    missing = query_mids - bindings.keys()
    if missing:
        raise ValueError(
            f"SPARQL MIDs lack WebQSP/RoG surface bindings: {sorted(missing)}"
        )
    return {mid: bindings[mid] for mid in sorted(query_mids)}


def _unresolved_parse(
    parse_id: str, phase: str, error: Exception
) -> dict[str, Any]:
    return {
        "parse_id": parse_id,
        "status": "unresolved",
        "phase": phase,
        "error": f"{type(error).__name__}: {error}",
    }


def validate_supported_sparql(sparql: object) -> str:
    if not isinstance(sparql, str) or not sparql.strip():
        raise ValueError("WebQSP parse lacks SPARQL")
    if len(re.findall(r"\bSELECT\b", sparql, re.IGNORECASE)) != 1:
        raise ValueError("nested SPARQL SELECT is unsupported")
    return sparql


def classify_row(
    rog_row: dict[str, Any],
    webqsp_question: dict[str, Any],
    supplemental_mid_names: dict[str, str],
    reverse_properties: dict[str, str],
) -> dict[str, Any]:
    example_id = str(rog_row.get("id", "")).strip()
    if not example_id:
        raise ValueError("RoG row lacks id")
    if str(webqsp_question.get("QuestionId", "")).strip() != example_id:
        raise ValueError(f"RoG and WebQSP IDs differ: {example_id}")

    rog_question = rog_row.get("question")
    official_question = webqsp_question.get("ProcessedQuestion")
    if not isinstance(rog_question, str) or not isinstance(
        official_question, str
    ):
        raise ValueError(f"question text is missing: {example_id}")
    if canonical_text(rog_question) != canonical_text(official_question):
        raise ValueError(f"question texts do not align: {example_id}")

    q_entities = validated_string_list(
        rog_row.get("q_entity"), field="q_entity", example_id=example_id
    )
    gold_answers = validated_string_list(
        rog_row.get("a_entity"), field="a_entity", example_id=example_id
    )
    graph = rog_row.get("graph")
    if not isinstance(graph, list):
        raise ValueError(f"RoG row lacks graph: {example_id}")

    parses = webqsp_question.get("Parses")
    if not isinstance(parses, list) or not parses:
        raise ValueError(f"WebQSP question lacks parses: {example_id}")

    evaluations: list[dict[str, Any]] = []
    parse_ids: set[str] = set()
    for parse_number, parse_record in enumerate(parses):
        if not isinstance(parse_record, dict):
            raise ValueError(f"WebQSP parse is not an object: {example_id}")
        if (parse_record.get("AnnotatorComment") or {}).get(
            "ParseQuality"
        ) != "Complete":
            continue
        parse_id = str(
            parse_record.get("ParseId", f"{example_id}.P{parse_number}")
        ).strip()
        if parse_id in parse_ids:
            raise ValueError(f"duplicate WebQSP ParseId: {parse_id}")
        parse_ids.add(parse_id)

        try:
            plan = parse_query(
                validate_supported_sparql(parse_record.get("Sparql"))
            )
        except ValueError as error:
            evaluations.append(_unresolved_parse(parse_id, "parse", error))
            continue
        try:
            mid_bindings = build_mid_bindings(
                parse_record, q_entities, supplemental_mid_names
            )
        except ValueError as error:
            evaluations.append(_unresolved_parse(parse_id, "binding", error))
            continue
        try:
            result = evaluate_query(
                plan,
                graph,
                mid_bindings,
                gold_answers,
                reverse_properties,
            )
        except (EvaluationLimit, ValueError) as error:
            evaluations.append(_unresolved_parse(parse_id, "evaluation", error))
            continue

        status = (
            "complete_witness"
            if result["has_complete_witness"]
            else "no_witness"
        )
        evaluations.append({
            "parse_id": parse_id,
            "status": status,
            "mid_bindings": mid_bindings,
            **result,
        })
        if result["has_complete_witness"]:
            return {
                "id": example_id,
                "classification": "complete_witness",
                "winning_parse_id": parse_id,
                "gold_answers": gold_answers,
                "parse_evaluations": evaluations,
            }

    if not evaluations:
        raise ValueError(f"WebQSP question has no complete parses: {example_id}")
    classification = (
        "unresolved"
        if any(item["status"] == "unresolved" for item in evaluations)
        else "no_witness"
    )
    return {
        "id": example_id,
        "classification": classification,
        "winning_parse_id": None,
        "gold_answers": gold_answers,
        "parse_evaluations": evaluations,
    }


def split_dataset(
    *,
    rog_rows: Iterable[dict[str, Any]],
    webqsp_path: Path,
    supplemental_mid_names_path: Path,
    reverse_properties_path: Path,
    output_dir: Path,
    rog_source: dict[str, Any],
) -> dict[str, Any]:
    """Atomically write K, U, unresolved, details, and a run manifest."""

    for path in (
        webqsp_path,
        supplemental_mid_names_path,
        reverse_properties_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"split input does not exist: {path}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    questions = load_webqsp_questions(webqsp_path)
    supplemental_mid_names = load_supplemental_mid_names(
        supplemental_mid_names_path
    )
    reverse_properties = load_reverse_properties(reverse_properties_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    filenames = {
        "complete_witness": "complete_witness.jsonl",
        "no_witness": "no_witness.jsonl",
        "unresolved": "unresolved.jsonl",
    }
    seen_ids: set[str] = set()
    counts = Counter()
    parse_status_counts = Counter()
    unresolved_phases = Counter()

    try:
        handles = {
            label: (temp_dir / filename).open(
                "x", encoding="utf-8", newline="\n"
            )
            for label, filename in filenames.items()
        }
        details = (temp_dir / "witness_details.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        )
        try:
            for row_number, row in enumerate(rog_rows, start=1):
                if not isinstance(row, dict):
                    raise ValueError(f"RoG row {row_number} is not an object")
                example_id = str(row.get("id", "")).strip()
                if not example_id:
                    raise ValueError(f"RoG row {row_number} lacks id")
                if example_id in seen_ids:
                    raise ValueError(f"duplicate RoG ID: {example_id}")
                if example_id not in questions:
                    raise ValueError(
                        f"RoG ID is absent from official WebQSP: {example_id}"
                    )
                seen_ids.add(example_id)

                detail = classify_row(
                    row,
                    questions[example_id],
                    supplemental_mid_names,
                    reverse_properties,
                )
                classification = detail["classification"]
                handles[classification].write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                details.write(
                    json.dumps(
                        {"row_number": row_number, **detail},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                counts[classification] += 1
                for evaluation in detail["parse_evaluations"]:
                    parse_status_counts[evaluation["status"]] += 1
                    if evaluation["status"] == "unresolved":
                        unresolved_phases[evaluation["phase"]] += 1
                if row_number % 100 == 0:
                    print(f"evaluated {row_number} RoG-WebQSP rows", flush=True)
        finally:
            details.close()
            for handle in handles.values():
                handle.close()

        count_payload = {
            "total": len(seen_ids),
            **{label: counts[label] for label in filenames},
        }
        if sum(count_payload[label] for label in filenames) != len(seen_ids):
            raise RuntimeError("WebQSP output counts do not form a partition")
        manifest = {
            "version": 2,
            "criterion": (
                "complete_witness iff at least one official complete WebQSP gold "
                "SPARQL parse returns a RoG a_entity on the directed folded graph "
                "plus schema-declared inverse closure with positive graph "
                "evidence for every answer-changing condition; absence-only "
                "NOT EXISTS branches do not establish evidence; no_witness iff "
                "every parse is evaluated and none succeeds; otherwise unresolved"
            ),
            "cvt_policy": (
                "An ordered SPARQL path through a removed CVT matches one folded "
                "predicate formed by joining its atomic predicates with '-'"
            ),
            "inverse_property_policy": (
                "Only inverse pairs declared by the supplied GrailQA "
                "reverse_properties file are used"
            ),
            "mid_binding_policy": (
                "Parse topics are matched by official TopicEntityName to RoG "
                "q_entity; constraint MIDs use official EntityName; remaining "
                "manual-query constants use the explicit supplemental MID map"
            ),
            "sources": {
                "rog": rog_source,
                "official_webqsp": str(webqsp_path.resolve()),
                "supplemental_mid_names": str(
                    supplemental_mid_names_path.resolve()
                ),
                "reverse_properties": str(reverse_properties_path.resolve()),
            },
            "counts": count_payload,
            "official_question_count": len(questions),
            "official_questions_absent_from_rog": len(questions) - len(seen_ids),
            "evaluated_parse_status_counts": dict(
                sorted(parse_status_counts.items())
            ),
            "unresolved_parse_phase_counts": dict(
                sorted(unresolved_phases.items())
            ),
            "outputs": {
                **filenames,
                "details": "witness_details.jsonl",
            },
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temp_dir.replace(output_dir)
        return manifest
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct WebQSP S/I splits by replaying official gold SPARQL on RoG input graphs."
    )
    parser.add_argument("--rog-jsonl", type=Path, required=True)
    parser.add_argument("--webqsp", type=Path, required=True)
    parser.add_argument("--supplemental-mid-names", type=Path, required=True)
    parser.add_argument("--reverse-properties", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = split_dataset(
        rog_rows=io.read_jsonl(args.rog_jsonl),
        webqsp_path=args.webqsp,
        supplemental_mid_names_path=args.supplemental_mid_names,
        reverse_properties_path=args.reverse_properties,
        output_dir=args.output_dir,
        rog_source={"kind": "jsonl", "path": str(args.rog_jsonl.resolve())},
    )
    print(json.dumps({
        "total": manifest["counts"]["total"],
        "S": manifest["counts"]["complete_witness"],
        "I": manifest["counts"]["no_witness"],
        "unresolved": manifest["counts"]["unresolved"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
