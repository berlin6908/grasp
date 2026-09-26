"""Resolve cited evidence and the registered semantic background."""

from __future__ import annotations

import re
from collections.abc import Iterable

from dataset_split.cwq import canonical_text
from src.answering.prompt import flatten_step_evidence

Triple = tuple[str, str, str]


SEMANTIC_BACKGROUND_RELATIONS = frozenset({
    "common.topic.notable_types",
})


EVIDENCE_ID = re.compile(r"\b(s\d+)\.t(\d+)\b", re.IGNORECASE)


EVIDENCE_RANGE = re.compile(
    r"\b(s\d+)\.t(\d+)\s*[-\u2010-\u2015]\s*(?:(s\d+)\.)?t(\d+)\b",
    re.IGNORECASE,
)


BRACKETED_CITATIONS = re.compile(r"\[([^\]]+)]")


FLAT_EVIDENCE_ID = re.compile(r"\be(\d+)\b", re.IGNORECASE)


FLAT_EVIDENCE_RANGE = re.compile(
    r"\be(\d+)\s*[-\u2010-\u2015]\s*e?(\d+)\b",
    re.IGNORECASE,
)


def normalize_triple(raw: object) -> Triple:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"malformed triple: {raw!r}")
    triple = tuple(str(part).strip() for part in raw)
    if any(not part for part in triple):
        raise ValueError(f"empty triple component: {raw!r}")
    return triple


def triple_key(triple: Triple) -> Triple:
    return (
        canonical_text(triple[0]),
        triple[1],
        canonical_text(triple[2]),
    )


def unique_triples(raw_triples: Iterable[object]) -> list[Triple]:
    triples: list[Triple] = []
    seen: set[Triple] = set()
    for raw in raw_triples:
        triple = normalize_triple(raw)
        key = triple_key(triple)
        if key not in seen:
            seen.add(key)
            triples.append(triple)
    return triples


def _unique_answers(raw_answers: Iterable[object]) -> list[str]:
    answers: list[str] = []
    seen: set[str] = set()
    for raw in raw_answers:
        answer = str(raw).strip()
        key = canonical_text(answer)
        if answer and key not in seen:
            seen.add(key)
            answers.append(answer)
    return answers


def graph_conditioned_answers(
    raw_answers: Iterable[object], graph: list[Triple],
) -> list[str]:
    graph_nodes = {
        canonical_text(node)
        for head, _relation, tail in graph
        for node in (head, tail)
    }
    return [
        answer
        for answer in _unique_answers(raw_answers)
        if canonical_text(answer) in graph_nodes
    ]


def semantic_evidence(
    cited_triples: list[Triple], graph: list[Triple],
) -> list[Triple]:
    return unique_triples([
        *cited_triples,
        *(
            triple
            for triple in graph
            if triple[1] in SEMANTIC_BACKGROUND_RELATIONS
        ),
    ])


def packed_evidence_by_id(
    row: dict,
    *,
    citation_namespace: str = "step",
) -> dict[str, Triple]:
    if citation_namespace == "flat":
        return {
            f"e{index}": normalize_triple(raw)
            for index, raw in enumerate(
                flatten_step_evidence(row["step_evidence"]), start=1,
            )
        }
    if citation_namespace != "step":
        raise ValueError(f"unknown citation namespace: {citation_namespace}")
    return {
        f"{step_id}.t{triple_index}".casefold(): normalize_triple(raw)
        for step_id, step in row["step_evidence"].items()
        if "triples" in step
        for triple_index, raw in enumerate(step["triples"], start=1)
    }


def extract_cited_evidence_ids(reason: str) -> set[str]:
    cited = {
        f"{step.casefold()}.t{number}"
        for step, number in EVIDENCE_ID.findall(reason)
    }
    for step, start_text, end_step, end_text in EVIDENCE_RANGE.findall(reason):
        step = step.casefold()
        if end_step and end_step.casefold() != step:
            continue
        start, end = int(start_text), int(end_text)
        if end < start:
            continue
        cited.update(f"{step}.t{number}" for number in range(start, end + 1))
    for bracket in BRACKETED_CITATIONS.findall(reason):
        cited.update(f"e{number}" for number in FLAT_EVIDENCE_ID.findall(bracket))
        for start_text, end_text in FLAT_EVIDENCE_RANGE.findall(bracket):
            start, end = int(start_text), int(end_text)
            if end >= start:
                cited.update(f"e{number}" for number in range(start, end + 1))
    return cited
