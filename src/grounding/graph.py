"""Normalize and index the input graph for directed relation execution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


Triple = list[str]


@dataclass(frozen=True)
class RelationIndex:
    forward: dict[tuple[str, str], list[Triple]]
    reverse: dict[tuple[str, str], list[Triple]]
    adjacent: dict[str, set[tuple[str, str]]]


def normalize_graph(graph: Iterable[Triple]) -> list[Triple]:
    output = []
    seen = set()
    for triple in graph:
        normalized = tuple(" ".join(part.split()) for part in triple)
        if normalized not in seen:
            seen.add(normalized)
            output.append(list(normalized))
    return output


def normalize_value(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def adjacent_relations(
    relation_index: RelationIndex, query_entities: list[str],
) -> set[tuple[str, str]]:
    return {
        signature
        for entity in query_entities
        for signature in relation_index.adjacent.get(normalize_value(entity), ())
    }


def build_relation_index(full_graph: list[Triple]) -> RelationIndex:
    forward: dict[tuple[str, str], list[Triple]] = defaultdict(list)
    reverse: dict[tuple[str, str], list[Triple]] = defaultdict(list)
    adjacent: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for triple in full_graph:
        head, relation, tail = triple
        head_key, tail_key = normalize_value(head), normalize_value(tail)
        copied = list(triple)
        forward[(head_key, relation)].append(copied)
        reverse[(tail_key, relation)].append(copied)
        adjacent[head_key].add((relation, "outgoing"))
        adjacent[tail_key].add((relation, "incoming"))
    return RelationIndex(forward, reverse, adjacent)


def expand_relation(
    *, relation_index: RelationIndex, seed: str, predicate: str, direction: str,
) -> list[tuple[str, list[Triple]]]:
    endpoints: dict[str, tuple[str, Triple]] = {}
    directed_index = relation_index.forward if direction == "outgoing" else relation_index.reverse
    for triple in directed_index.get((normalize_value(seed), predicate), ()):
        value = triple[2] if direction == "outgoing" else triple[0]
        key = normalize_value(value)
        current = endpoints.get(key)
        if current is None or tuple(triple) < tuple(current[1]):
            endpoints[key] = (value, triple)
    return [(value, [list(triple)]) for _key, (value, triple) in sorted(endpoints.items())]
