#!/usr/bin/env python
"""Split RoG-CWQ by executing original CWQ SPARQL on each folded subgraph.

RoG represents a path through a removed CVT node as one directed edge whose
predicate is the atomic Freebase predicates joined by ``-``.  This follows the
RoG dataset construction and the NSM Freebase preprocessing it cites:

https://github.com/RManLuo/reasoning-on-graphs
https://github.com/RichardHGL/WSDM2021_NSM/tree/737f8e867af822179388a0846e6d81749c5bece3/preprocessing/Freebase

The graph is also closed under schema-declared Freebase inverse properties.
The mapping is supplied explicitly from GrailQA's processed Freebase ontology:

https://github.com/dki-lab/GrailQA/blob/bc15df916ca4101f773722151c90ba3f9eff9df5/ontology/reverse_properties

The original SPARQL MIDs are bound to RoG surface nodes with the same ordered
topic-entity records from NSM.  They are never guessed from SPARQL occurrence
order or treated as interchangeable question entities.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from src import io
from src.io import sha256_file


TERM = r'(?:\?[A-Za-z]\w*|ns:[A-Za-z0-9_.]+|"(?:[^"\\]|\\.)*"(?:\^\^xsd:\w+)?|[-+]?\d+(?:\.\d+)?)'
TOKEN_RE = re.compile(rf"{TERM}|[{{}}.;,]")
RELATION_RE = re.compile(r"^ns:(?![mg]\.)[A-Za-z0-9_.]+$")
COMPARISON_RE = re.compile(r"^(.+?)\s*(<=|>=|!=|=|<|>)\s*(.+?)\s*\.?$", re.DOTALL)
MID_RE = re.compile(r"ns:([mg]\.[A-Za-z0-9_]+)")
MAXIMUM_WITNESSES = 100_000
GRAILQA_REVERSE_PROPERTIES_URL = (
    "https://raw.githubusercontent.com/dki-lab/GrailQA/"
    "bc15df916ca4101f773722151c90ba3f9eff9df5/ontology/reverse_properties"
)


@dataclass(frozen=True)
class Pattern:
    index: int
    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class Match:
    covered: tuple[int, ...]
    start: str
    end: str
    graph_index: int
    inverse: bool


@dataclass(frozen=True)
class Operand:
    term: str
    cast: str | None = None


@dataclass(frozen=True)
class Comparison:
    left: Operand
    operator: str
    right: Operand
    source: str


@dataclass(frozen=True)
class OptionalExists:
    subject: str
    relation: str
    comparison: Comparison
    source: str


@dataclass(frozen=True)
class Order:
    variable: str
    cast: str | None
    descending: bool


@dataclass(frozen=True)
class QueryPlan:
    branches: tuple[tuple[Pattern, ...], ...]
    comparisons: tuple[Comparison, ...]
    optional_exists: tuple[OptionalExists, ...]
    order: Order | None
    answer_variable: str


class EvaluationLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class NsmRow:
    question: str
    topic_indices: tuple[int, ...]
    answers: tuple[str, ...]


def canonical_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.split()).casefold()


def align_mid_constants(
    sparql: str,
    nsm_topic_mids: list[str],
    rog_q_entities: list[str],
) -> dict[str, str]:
    """Bind every SPARQL MID using NSM topic order and RoG topic surfaces."""

    if len(nsm_topic_mids) != len(rog_q_entities):
        raise ValueError(
            "NSM topic MID count differs from RoG q_entity count: "
            f"{len(nsm_topic_mids)} != {len(rog_q_entities)}"
        )
    if len(set(nsm_topic_mids)) != len(nsm_topic_mids):
        raise ValueError("NSM topic MID list contains duplicates")
    if any(not isinstance(value, str) or not value.strip() for value in rog_q_entities):
        raise ValueError("RoG q_entity values must be non-empty strings")

    query_mids = set(MID_RE.findall(sparql))
    topic_mids = set(nsm_topic_mids)
    if query_mids != topic_mids:
        raise ValueError(
            "SPARQL MID set differs from the NSM topic MID set: "
            f"query_only={sorted(query_mids - topic_mids)}, "
            f"nsm_only={sorted(topic_mids - query_mids)}"
        )
    return dict(zip(nsm_topic_mids, rog_q_entities, strict=True))


def strip_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        quote = False
        escaped = False
        kept: list[str] = []
        for char in line:
            if quote:
                kept.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quote = False
            elif char == '"':
                quote = True
                kept.append(char)
            elif char == "#":
                break
            else:
                kept.append(char)
        lines.append("".join(kept))
    return "\n".join(lines)


def balanced_end(text: str, start: int, opening: str, closing: str) -> int:
    if start >= len(text) or text[start] != opening:
        raise ValueError(f"expected {opening!r} at offset {start}")
    depth = 0
    quote = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
            continue
        if char == '"':
            quote = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError(f"unbalanced {opening}{closing} block")


def where_body(sparql: str) -> tuple[str, str]:
    clean = strip_comments(sparql)
    match = re.search(r"\bWHERE\b", clean, re.IGNORECASE)
    if not match:
        raise ValueError("SPARQL has no WHERE clause")
    start = clean.find("{", match.end())
    if start < 0:
        raise ValueError("SPARQL WHERE clause has no body")
    end = balanced_end(clean, start, "{", "}")
    return clean[start + 1 : end - 1], clean[end:]


def filter_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    start_re = re.compile(r"\bFILTER\b", re.IGNORECASE)
    while match := start_re.search(text, cursor):
        opening = text.find("(", match.end())
        if opening < 0:
            raise ValueError("FILTER without parenthesis")
        end = balanced_end(text, opening, "(", ")")
        spans.append((match.start(), end, text[opening + 1 : end - 1].strip()))
        cursor = end
    return spans


def remove_filters(text: str) -> str:
    for start, end, _ in reversed(filter_spans(text)):
        text = text[:start] + " " + text[end:]
    return text


def split_union(body: str) -> list[str]:
    clean = remove_filters(body)
    matches = list(re.finditer(r"\bUNION\b", clean, re.IGNORECASE))
    if not matches:
        return [clean]
    if len(matches) != 1:
        raise ValueError(f"unsupported UNION count: {len(matches)}")
    union = matches[0]
    left_close = union.start() - 1
    while left_close >= 0 and clean[left_close].isspace():
        left_close -= 1
    right_open = union.end()
    while right_open < len(clean) and clean[right_open].isspace():
        right_open += 1
    if (
        left_close < 0
        or clean[left_close] != "}"
        or right_open >= len(clean)
        or clean[right_open] != "{"
    ):
        raise ValueError("UNION alternatives are not braced")

    depth = 0
    left_open: int | None = None
    for index in range(left_close, -1, -1):
        if clean[index] == "}":
            depth += 1
        elif clean[index] == "{":
            depth -= 1
            if depth == 0:
                left_open = index
                break
    if left_open is None:
        raise ValueError("unbalanced left UNION branch")
    right_end = balanced_end(clean, right_open, "{", "}")
    prefix = clean[:left_open]
    suffix = clean[right_end:]
    return [
        prefix + clean[left_open + 1 : left_close] + suffix,
        prefix + clean[right_open + 1 : right_end - 1] + suffix,
    ]


def is_term(token: str) -> bool:
    return bool(re.fullmatch(TERM, token))


def parse_basic_patterns(text: str) -> list[Pattern]:
    tokens = TOKEN_RE.findall(text)
    patterns: list[Pattern] = []
    subject: str | None = None
    predicate: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            is_term(token)
            and index + 2 < len(tokens)
            and RELATION_RE.match(tokens[index + 1])
            and is_term(tokens[index + 2])
        ):
            subject, predicate, obj = token, tokens[index + 1][3:], tokens[index + 2]
            patterns.append(Pattern(len(patterns) + 1, subject, predicate, obj))
            index += 3
        elif (
            token == ";"
            and subject
            and index + 2 < len(tokens)
            and RELATION_RE.match(tokens[index + 1])
            and is_term(tokens[index + 2])
        ):
            predicate, obj = tokens[index + 1][3:], tokens[index + 2]
            patterns.append(Pattern(len(patterns) + 1, subject, predicate, obj))
            index += 3
        elif (
            token == ","
            and subject
            and predicate
            and index + 1 < len(tokens)
            and is_term(tokens[index + 1])
        ):
            obj = tokens[index + 1]
            patterns.append(Pattern(len(patterns) + 1, subject, predicate, obj))
            index += 2
        else:
            if token in {".", "}"}:
                subject = predicate = None
            index += 1
    if not patterns:
        raise ValueError("query branch has no triple patterns")
    return patterns


def parse_operand(text: str) -> Operand:
    text = text.strip()
    cast = re.fullmatch(
        r"xsd:(integer|datetime|float)\s*\(\s*(.+?)\s*\)",
        text,
        re.IGNORECASE,
    )
    if cast:
        return Operand(cast.group(2), cast.group(1).lower())
    if not re.fullmatch(TERM, text):
        raise ValueError(f"unsupported operand: {text!r}")
    return Operand(text)


def parse_comparison(text: str) -> Comparison:
    difference = re.fullmatch(
        r"xsd:datetime\((\?\w+)\)\s*-\s*xsd:datetime\((\?\w+)\)\s*<\s*0\s*\.?",
        text.strip(),
        re.IGNORECASE,
    )
    if difference:
        return Comparison(
            Operand(difference.group(1), "datetime"),
            "<",
            Operand(difference.group(2), "datetime"),
            text,
        )
    match = COMPARISON_RE.fullmatch(" ".join(text.split()))
    if not match:
        raise ValueError(f"unsupported FILTER comparison: {text!r}")
    return Comparison(
        parse_operand(match.group(1)),
        match.group(2),
        parse_operand(match.group(3)),
        text,
    )


def parse_optional_exists(text: str) -> OptionalExists:
    triples = list(
        re.finditer(
            rf"(?P<s>{TERM})\s+ns:(?P<r>[A-Za-z0-9_.]+)\s+(?P<o>{TERM})\s*\.?",
            text,
        )
    )
    if len(triples) != 2:
        raise ValueError(
            f"optional EXISTS filter has {len(triples)} triples: {text!r}"
        )
    first, second = triples
    if first["s"] != second["s"] or first["r"] != second["r"]:
        raise ValueError(f"NOT EXISTS and EXISTS paths differ: {text!r}")
    nested = re.search(r"FILTER\s*\((.+?)\)\s*\}", text, re.IGNORECASE | re.DOTALL)
    if not nested:
        raise ValueError(f"optional EXISTS filter lacks comparison: {text!r}")
    return OptionalExists(
        first["s"],
        first["r"],
        parse_comparison(nested.group(1)),
        text,
    )


def parse_filters(body: str) -> tuple[list[Comparison], list[OptionalExists]]:
    comparisons: list[Comparison] = []
    optional: list[OptionalExists] = []
    for _, _, expression in filter_spans(body):
        upper = expression.upper()
        if upper.startswith("!ISLITERAL"):
            continue
        if upper.startswith("NOT EXISTS"):
            optional.append(parse_optional_exists(expression))
            continue
        string_match = re.fullmatch(
            r"str\((\?\w+)\)\s*=\s*(.+)",
            expression.strip(),
            re.IGNORECASE | re.DOTALL,
        )
        if string_match:
            comparisons.append(
                Comparison(
                    Operand(string_match.group(1), "str"),
                    "=",
                    parse_operand(string_match.group(2)),
                    expression,
                )
            )
            continue
        comparisons.append(parse_comparison(expression))
    return comparisons, optional


def parse_order(tail: str) -> Order | None:
    order_match = re.search(
        r"ORDER\s+BY\s+(?:(ASC|DESC)\s*\()?\s*"
        r"(?:(xsd:(integer|datetime|float))\s*\()?\s*"
        r"(\?\w+)\s*\)?\s*\)?",
        tail,
        re.IGNORECASE,
    )
    limit_match = re.search(r"\bLIMIT\s+(\d+)", tail, re.IGNORECASE)
    if not order_match:
        if limit_match:
            raise ValueError("LIMIT without ORDER BY")
        return None
    limit = int(limit_match.group(1)) if limit_match else 1
    if limit != 1:
        raise ValueError(f"unsupported LIMIT {limit}")
    return Order(
        order_match.group(4),
        order_match.group(3).lower() if order_match.group(3) else None,
        (order_match.group(1) or "").casefold() == "desc",
    )


def parse_query(sparql: str) -> QueryPlan:
    select = re.search(
        r"\bSELECT\s+(?:DISTINCT\s+)?(\?\w+)",
        strip_comments(sparql),
        re.IGNORECASE,
    )
    if not select:
        raise ValueError("SPARQL SELECT variable is unsupported")
    body, tail = where_body(sparql)
    comparisons, optional = parse_filters(body)
    branches = tuple(
        tuple(parse_basic_patterns(branch)) for branch in split_union(body)
    )
    return QueryPlan(
        branches,
        tuple(comparisons),
        tuple(optional),
        parse_order(tail),
        select.group(1),
    )


def relation_paths(
    patterns: tuple[Pattern, ...],
) -> dict[tuple[str, ...], list[tuple[Pattern, ...]]]:
    outgoing: dict[str, list[Pattern]] = defaultdict(list)
    for pattern in patterns:
        outgoing[pattern.subject].append(pattern)
    paths: dict[tuple[str, ...], list[tuple[Pattern, ...]]] = defaultdict(list)

    def visit(path: tuple[Pattern, ...]) -> None:
        paths[tuple(item.relation for item in path)].append(path)
        if not path[-1].object.startswith("?"):
            return
        used = {item.index for item in path}
        for following in outgoing.get(path[-1].object, []):
            if following.index not in used:
                visit((*path, following))

    for pattern in patterns:
        visit((pattern,))
    return paths


def inverse_relation_path(
    relations: tuple[str, ...],
    reverse_properties: dict[str, str],
) -> tuple[str, ...] | None:
    if any(relation not in reverse_properties for relation in relations):
        return None
    return tuple(reverse_properties[relation] for relation in reversed(relations))


def candidate_matches(
    patterns: tuple[Pattern, ...],
    graph: list[list[str]],
    reverse_properties: dict[str, str],
    relation_matches: dict[str, frozenset[str]] | None = None,
    answer_variable: str | None = None,
) -> list[Match]:
    paths = relation_paths(patterns)
    paths = {
        key: [path for path in group
              if all(item.object != answer_variable for item in path[:-1])]
        for key, group in paths.items()
    }
    inverse_paths: dict[tuple[str, ...], list[tuple[Pattern, ...]]] = defaultdict(list)
    for relation_key, query_paths in paths.items():
        inverse_key = inverse_relation_path(relation_key, reverse_properties)
        if inverse_key is not None:
            inverse_paths[inverse_key].extend(query_paths)
    matches: list[Match] = []
    seen: set[tuple[tuple[int, ...], str, str, int, bool]] = set()
    groups: dict[tuple[str, ...], tuple[list, list]] = {}
    for graph_index, (_head, relation, _tail) in enumerate(graph, start=1):
        relation_key = tuple(relation.split("-"))
        if relation_key not in groups:
            groups[relation_key] = tuple(
                [path for key in matching_relation_keys(relation_key, choices, relation_matches)
                 for path in choices[key]]
                for choices in (paths, inverse_paths)
            )
        for inverse, path_group in (
            (False, groups[relation_key][0]),
            (True, groups[relation_key][1]),
        ):
            for path in path_group:
                covered = tuple(item.index for item in path)
                start = path[0].subject
                end = path[-1].object
                key = (covered, start, end, graph_index, inverse)
                if key not in seen:
                    seen.add(key)
                    matches.append(Match(covered, start, end, graph_index, inverse))
    return sorted(matches, key=lambda match: (match.inverse, match.graph_index))


def matching_relation_keys(
    query: tuple[str, ...],
    available: dict,
    relation_matches: dict[str, frozenset[str]] | None,
) -> list[tuple[str, ...]]:
    """Match ordered atomic paths; None selects the existing exact condition."""
    if relation_matches is None:
        return [query] if query in available else []
    return [
        path for path in available
        if len(path) == len(query)
        and all(left == right or right in relation_matches[left]
                for left, right in zip(query, path))
    ]


def literal_value(term: str) -> str | None:
    if term.startswith('"'):
        return term[1 : term.rfind('"')]
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", term):
        return term
    return None


def expected_constant(term: str, mid_nodes: dict[str, str]) -> str | None:
    literal = literal_value(term)
    if literal is not None:
        return literal
    if re.fullmatch(r"ns:[mg]\.[A-Za-z0-9_]+", term):
        mid = term[3:]
        if mid not in mid_nodes:
            raise ValueError(f"SPARQL MID has no NSM/RoG binding: {mid}")
        return mid_nodes[mid]
    if term.startswith("ns:"):
        return term.rsplit(".", 1)[-1].replace("_", " ")
    return None


def bind_term(
    term: str,
    node: str,
    bindings: dict[str, str],
    mid_nodes: dict[str, str],
    gold_keys: set[str] | None,
    answer_variable: str,
) -> dict[str, str] | None:
    if term.startswith("?"):
        if (
            term == answer_variable
            and gold_keys is not None
            and canonical_text(node) not in gold_keys
        ):
            return None
        if term in bindings and canonical_text(bindings[term]) != canonical_text(node):
            return None
    else:
        expected = expected_constant(term, mid_nodes)
        if expected is None or canonical_text(expected) != canonical_text(node):
            return None
    updated = dict(bindings)
    updated[term] = node
    return updated


def iter_witnesses(
    patterns: tuple[Pattern, ...],
    matches: list[Match],
    graph: list[list[str]],
    mid_nodes: dict[str, str],
    gold_answers: list[str] | None,
    answer_variable: str,
    comparisons: tuple[Comparison, ...] = (),
    order: Order | None = None,
) -> Iterator[tuple[list[Match], dict[str, str]]]:
    gold_keys = (
        None
        if gold_answers is None
        else {canonical_text(value) for value in gold_answers}
    )

    def bind_match(match: Match, bindings: dict[str, str]) -> dict[str, str] | None:
        head, _relation, tail = graph[match.graph_index - 1]
        start_node, end_node = (tail, head) if match.inverse else (head, tail)
        updated = bind_term(match.start, start_node, bindings, mid_nodes, gold_keys, answer_variable)
        if updated is None:
            return None
        return bind_term(match.end, end_node, updated, mid_nodes, gold_keys, answer_variable)

    by_pattern: dict[int, dict[tuple[str, str], list[int]]] = defaultdict(dict)
    by_start: dict[tuple[int, str, str, str], list[int]] = defaultdict(list)
    by_end: dict[tuple[int, str, str, str], list[int]] = defaultdict(list)
    endpoints: dict[int, tuple[str, str]] = {}
    for position, match in enumerate(matches):
        # Apply known constants and credited-answer restrictions before choosing
        # a join order, so unrelated graph edges do not dominate candidate counts.
        if bind_match(match, {}) is None:
            continue
        head, _relation, tail = graph[match.graph_index - 1]
        start_node, end_node = (tail, head) if match.inverse else (head, tail)
        start_key, end_key = canonical_text(start_node), canonical_text(end_node)
        endpoints[position] = start_key, end_key
        for pattern_index in match.covered:
            by_pattern[pattern_index].setdefault((match.start, match.end), []).append(position)
            by_start[pattern_index, match.start, match.end, start_key].append(position)
            by_end[pattern_index, match.start, match.end, end_key].append(position)

    def candidates(pattern_index: int, bindings: dict[str, str]) -> list[int]:
        eligible = []
        for (start, end), group in by_pattern[pattern_index].items():
            start_key = canonical_text(bindings[start]) if start in bindings else None
            end_key = canonical_text(bindings[end]) if end in bindings else None
            options = [group]
            if start_key is not None:
                options.append(by_start.get((pattern_index, start, end, start_key), []))
            if end_key is not None:
                options.append(by_end.get((pattern_index, start, end, end_key), []))
            for position in min(options, key=len):
                left, right = endpoints[position]
                if (start_key is None or left == start_key) and (end_key is None or right == end_key):
                    eligible.append(position)
        return sorted(eligible)

    seen: set[tuple[frozenset[int], tuple[tuple[str, str], ...]]] = set()
    yielded = 0

    def search(
        covered: frozenset[int],
        bindings: dict[str, str],
        chosen: list[Match],
    ) -> Iterator[tuple[list[Match], dict[str, str]]]:
        nonlocal yielded
        state = (
            covered,
            tuple(
                sorted(
                    (key, canonical_text(value))
                    for key, value in bindings.items()
                )
            ),
        )
        if state in seen:
            return
        seen.add(state)
        if len(covered) == len(patterns):
            # A folded path can hide internal variables. A witness must expose
            # the answer and every value needed by the retained query operators.
            if answer_variable not in bindings:
                return
            if order is not None and order.variable not in bindings:
                return
            if any(comparison_passes(item, bindings) is not True for item in comparisons):
                return
            yielded += 1
            if yielded > MAXIMUM_WITNESSES:
                raise EvaluationLimit(
                    f"query produced more than {MAXIMUM_WITNESSES} witnesses"
                )
            yield chosen, bindings
            return
        remaining = [
            pattern for pattern in patterns if pattern.index not in covered
        ]
        available = min(
            ((pattern, candidates(pattern.index, bindings)) for pattern in remaining),
            key=lambda item: (
                len(item[1]),
                -int(
                    item[0].subject in bindings
                    or item[0].object in bindings
                    or not item[0].subject.startswith("?")
                    or not item[0].object.startswith("?")
                    or item[0].subject == answer_variable
                    or item[0].object == answer_variable
                ),
            ),
        )[1]
        for position in available:
            match = matches[position]
            updated = bind_match(match, bindings)
            if updated is None:
                continue
            if any(
                operand_raw(item.left, updated) is not None
                and operand_raw(item.right, updated) is not None
                and comparison_passes(item, updated) is not True
                for item in comparisons
            ):
                continue
            if order is not None and order.variable in updated:
                if scalar(updated[order.variable], order.cast) is None:
                    continue
            yield from search(
                covered | frozenset(match.covered),
                updated,
                [*chosen, match],
            )

    yield from search(frozenset(), {}, [])


def operand_raw(operand: Operand, bindings: dict[str, str]) -> str | None:
    literal = literal_value(operand.term)
    if literal is not None:
        return literal
    return bindings.get(operand.term)


def scalar(value: str, cast: str | None) -> tuple[int, Any] | None:
    text = str(value).strip().strip('"')
    if cast in {"integer", "float"}:
        try:
            return 0, float(text.replace(",", ""))
        except ValueError:
            return None
    if cast == "datetime" or re.fullmatch(
        r"-?\d{1,4}(?:-\d{1,2}(?:-\d{1,2})?)?(?:T.*)?", text
    ):
        match = re.match(
            r"^(-?\d{1,4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", text
        )
        if not match:
            return None
        return 1, (
            int(match.group(1)),
            int(match.group(2) or 1),
            int(match.group(3) or 1),
        )
    try:
        return 0, float(text.replace(",", ""))
    except ValueError:
        return 2, canonical_text(text)


def comparison_passes(
    comparison: Comparison, bindings: dict[str, str]
) -> bool | None:
    left_raw = operand_raw(comparison.left, bindings)
    right_raw = operand_raw(comparison.right, bindings)
    if left_raw is None or right_raw is None:
        return None
    if comparison.left.cast == "str":
        left = (2, str(left_raw))
        right = (2, str(right_raw))
    elif (
        comparison.operator in {"=", "!="}
        and comparison.left.cast is None
        and comparison.right.cast is None
    ):
        left = (2, canonical_text(str(left_raw).strip('"')))
        right = (2, canonical_text(str(right_raw).strip('"')))
    else:
        cast = comparison.left.cast or comparison.right.cast
        left = scalar(left_raw, cast)
        right = scalar(right_raw, cast)
    if left is None or right is None or left[0] != right[0]:
        return None
    operations = {
        "<": lambda: left[1] < right[1],
        ">": lambda: left[1] > right[1],
        "<=": lambda: left[1] <= right[1],
        ">=": lambda: left[1] >= right[1],
        "=": lambda: left[1] == right[1],
        "!=": lambda: left[1] != right[1],
    }
    return operations[comparison.operator]()


def paths_to_term(
    patterns: tuple[Pattern, ...],
    bindings: dict[str, str],
    target: str,
) -> list[tuple[str, tuple[str, ...]]]:
    outgoing: dict[str, list[Pattern]] = defaultdict(list)
    for pattern in patterns:
        outgoing[pattern.subject].append(pattern)
    found: list[tuple[str, tuple[str, ...]]] = []
    for start in bindings:
        if start == target:
            found.append((start, ()))

        def visit(
            term: str,
            relations: tuple[str, ...],
            used: frozenset[int],
        ) -> None:
            for pattern in outgoing.get(term, []):
                if pattern.index in used:
                    continue
                updated = (*relations, pattern.relation)
                if pattern.object == target:
                    found.append((start, updated))
                elif pattern.object.startswith("?"):
                    visit(
                        pattern.object,
                        updated,
                        used | {pattern.index},
                    )

        visit(start, (), frozenset())
    return found


def optional_passes(
    optional: OptionalExists,
    patterns: tuple[Pattern, ...],
    bindings: dict[str, str],
    graph_index: dict[str, dict[tuple[str, ...], list[Match]]],
    relation_matches: dict[str, frozenset[str]] | None = None,
) -> tuple[bool | None, Match | None]:
    paths = paths_to_term(patterns, bindings, optional.subject)
    if not paths:
        return None, None
    matches: list[Match] = []
    for start, path in paths:
        available = graph_index.get(canonical_text(bindings[start]), {})
        for key in matching_relation_keys((*path, optional.relation), available, relation_matches):
            matches.extend(available[key])
    if not matches:
        return False, None
    for match in matches:
        updated = dict(bindings)
        updated[optional.comparison.left.term] = match.end
        outcome = comparison_passes(optional.comparison, updated)
        if outcome is True:
            return True, match
        # Ill-typed expression inputs reject this solution, not the whole query.
        # https://www.w3.org/TR/sparql11-query/#defn_algFilter
    return False, None


def witness_passes(
    plan: QueryPlan,
    branch: tuple[Pattern, ...],
    bindings: dict[str, str],
    graph_index: dict[str, dict[tuple[str, ...], list[Match]]],
    relation_matches: dict[str, frozenset[str]] | None = None,
) -> tuple[bool, str | None, list[Match]]:
    for comparison in plan.comparisons:
        outcome = comparison_passes(comparison, bindings)
        if outcome is None:
            return False, f"unresolved comparison: {comparison.source}", []
        if not outcome:
            return False, None, []
    condition_evidence: list[Match] = []
    for optional in plan.optional_exists:
        outcome, evidence = optional_passes(
            optional, branch, bindings, graph_index, relation_matches
        )
        if outcome is None:
            return False, f"unresolved optional filter: {optional.source}", []
        if not outcome:
            return False, None, []
        if evidence is None:
            raise RuntimeError("passing optional filter lacks positive evidence")
        condition_evidence.append(evidence)
    return True, None, condition_evidence


def evidence_indices(chosen: list[Match]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for match in chosen:
        if match.graph_index not in seen:
            seen.add(match.graph_index)
            result.append(match.graph_index)
    return result


def inverse_evidence_indices(chosen: list[Match]) -> list[int]:
    return evidence_indices([match for match in chosen if match.inverse])


def matching_gold(value: str, gold_answers: list[str]) -> str | None:
    key = canonical_text(value)
    return next(
        (answer for answer in gold_answers if canonical_text(answer) == key),
        None,
    )


def failed_evaluation(reason: str) -> dict[str, Any]:
    return {
        "has_complete_witness": False,
        "failure_reason": reason,
        "matched_gold_answer": None,
        "witness_graph_indices": [],
        "inverse_witness_graph_indices": [],
    }


def successful_evaluation(
    answer: str, chosen: list[Match]
) -> dict[str, Any]:
    return {
        "has_complete_witness": True,
        "failure_reason": None,
        "matched_gold_answer": answer,
        "witness_graph_indices": evidence_indices(chosen),
        "inverse_witness_graph_indices": inverse_evidence_indices(chosen),
    }


def evaluate_query(
    plan: QueryPlan,
    graph: list[list[str]],
    mid_nodes: dict[str, str],
    gold_answers: list[str],
    reverse_properties: dict[str, str],
    relation_matches: dict[str, frozenset[str]] | None = None,
) -> dict[str, Any]:
    """Return whether the folded graph sufficiently supports a gold answer."""

    if not gold_answers or any(
        not isinstance(answer, str) or not answer.strip()
        for answer in gold_answers
    ):
        raise ValueError("gold answers must be a non-empty list of strings")
    normalized_graph: list[list[str]] = []
    for index, triple in enumerate(graph, start=1):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            raise ValueError(f"malformed graph triple {index}")
        normalized = [str(value).strip() for value in triple]
        if any(not value for value in normalized):
            raise ValueError(f"empty graph triple value at {index}")
        normalized_graph.append(normalized)

    graph_index: dict[str, dict[tuple[str, ...], list[Match]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (head, relation, tail) in enumerate(normalized_graph, start=1):
        relation_path = tuple(relation.split("-"))
        graph_index[canonical_text(head)][relation_path].append(
            Match((), head, tail, index, False)
        )
        inverse_path = inverse_relation_path(relation_path, reverse_properties)
        if inverse_path is not None:
            graph_index[canonical_text(tail)][inverse_path].append(
                Match((), tail, head, index, True)
            )
    unresolved: set[str] = set()

    if plan.order is None:
        for branch in plan.branches:
            matches = candidate_matches(branch, normalized_graph, reverse_properties, relation_matches, plan.answer_variable)
            for chosen, bindings in iter_witnesses(
                branch,
                matches,
                normalized_graph,
                mid_nodes,
                gold_answers,
                plan.answer_variable,
                plan.comparisons,
                plan.order,
            ):
                passed, error, condition_evidence = witness_passes(
                    plan,
                    branch,
                    bindings,
                    graph_index,
                    relation_matches,
                )
                if error:
                    unresolved.add(error)
                if passed:
                    answer = matching_gold(
                        bindings.get(plan.answer_variable, ""), gold_answers
                    )
                    if answer is None:
                        raise ValueError(
                            "witness SELECT variable is not a gold answer"
                        )
                    return successful_evaluation(
                        answer, [*chosen, *condition_evidence]
                    )
        if unresolved:
            raise ValueError("; ".join(sorted(unresolved)))
        graph_nodes = {
            canonical_text(node)
            for head, _relation, tail in normalized_graph
            for node in (head, tail)
        }
        if not any(canonical_text(answer) in graph_nodes for answer in gold_answers):
            return failed_evaluation("gold_node_absent")
        return failed_evaluation("no_complete_witness")

    branch_matches = [
        (branch, candidate_matches(branch, normalized_graph, reverse_properties, relation_matches, plan.answer_variable))
        for branch in plan.branches
    ]

    def order_value(match: Match) -> str | None:
        head, _relation, tail = normalized_graph[match.graph_index - 1]
        if match.start == plan.order.variable:
            return tail if match.inverse else head
        if match.end == plan.order.variable:
            return head if match.inverse else tail
        return None

    values_by_key: dict[tuple[int, Any], set[str]] = defaultdict(set)
    for _branch, matches in branch_matches:
        for match in matches:
            raw = order_value(match)
            if raw is not None:
                key = scalar(raw, plan.order.cast)
                if key is not None:
                    values_by_key[key].add(canonical_text(raw))

    # ORDER BY ... LIMIT 1 only needs the first feasible key. At that key,
    # search for a gold solution before an arbitrary solution to retain ties.
    for key in sorted(values_by_key, reverse=plan.order.descending):
        allowed_values = values_by_key[key]
        restricted = [
            (branch, [match for match in matches
                      if order_value(match) is None
                      or canonical_text(order_value(match)) in allowed_values])
            for branch, matches in branch_matches
        ]
        for targets in (gold_answers, None):
            for branch, matches in restricted:
                for chosen, bindings in iter_witnesses(
                    branch, matches, normalized_graph, mid_nodes, targets,
                    plan.answer_variable, plan.comparisons, plan.order,
                ):
                    passed, error, condition_evidence = witness_passes(
                        plan, branch, bindings, graph_index, relation_matches,
                    )
                    if error:
                        unresolved.add(error)
                    if passed:
                        if targets is None:
                            return failed_evaluation("order_gold_not_selected")
                        answer = matching_gold(bindings[plan.answer_variable], gold_answers)
                        if answer is None:
                            raise ValueError("witness SELECT variable is not a gold answer")
                        return successful_evaluation(answer, [*chosen, *condition_evidence])
    if unresolved:
        raise ValueError("; ".join(sorted(unresolved)))
    return failed_evaluation("no_complete_witness")


def load_reverse_properties(path: Path) -> dict[str, str]:
    """Load Freebase inverse-property pairs as a symmetric mapping."""

    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2 or any(
                not re.fullmatch(r"[A-Za-z0-9_.-]+", field) for field in fields
            ):
                raise ValueError(
                    f"malformed reverse_properties line {line_number}: {path}"
                )
            left, right = fields
            for relation, inverse in ((left, right), (right, left)):
                previous = mapping.get(relation)
                if previous is not None and previous != inverse:
                    raise ValueError(
                        "conflicting inverse properties for "
                        f"{relation}: {previous} versus {inverse}"
                    )
                mapping[relation] = inverse
    if not mapping:
        raise ValueError(f"reverse_properties file is empty: {path}")
    return mapping


def load_original_cwq(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_id = io.index_rows(rows, "ID", "CWQ")
    for row in by_id.values():
        parse_query(row["sparql"])
    return by_id


def nsm_answer_surface(answer: object) -> str:
    """Read NSM entity labels and string-literal answers from their native fields."""

    if not isinstance(answer, dict):
        raise ValueError("NSM answer is not an object")
    text = answer.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    kb_id = answer.get("kb_id")
    if isinstance(kb_id, str) and kb_id.strip():
        return kb_id.strip()
    raise ValueError("NSM answer has neither display text nor a string kb_id")


def load_nsm_rows(path: Path) -> tuple[dict[str, NsmRow], set[int]]:
    by_id: dict[str, NsmRow] = {}
    needed_entity_indices: set[int] = set()
    for example_id, row in io.index_rows(io.read_jsonl(path), "id", "NSM").items():
        topic_indices = row["entities"]
        if len(set(topic_indices)) != len(topic_indices):
            raise ValueError(f"NSM row has duplicate entity indices: {example_id}")
        by_id[example_id] = NsmRow(
            question=row["question"],
            topic_indices=tuple(topic_indices),
            answers=tuple(nsm_answer_surface(answer) for answer in row["answers"]),
        )
        needed_entity_indices.update(topic_indices)
    return by_id, needed_entity_indices


def load_needed_entity_mids(
    path: Path, needed_indices: set[int]
) -> dict[int, str]:
    result: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index not in needed_indices:
                continue
            mid = line.strip()
            if not re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", mid):
                raise ValueError(f"NSM topic entity {index} is not a MID: {mid!r}")
            result[index] = mid
            if len(result) == len(needed_indices):
                break
    missing = needed_indices - result.keys()
    if missing:
        raise ValueError(
            "NSM entities.txt lacks required indices: "
            f"{sorted(missing)[:10]}"
        )
    return result


def validated_string_list(
    raw: object, *, field: str, example_id: str
) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"RoG row lacks {field}: {example_id}")
    values = [value.strip() for value in raw if isinstance(value, str) and value.strip()]
    if len(values) != len(raw):
        raise ValueError(f"RoG row has malformed {field}: {example_id}")
    return values


def same_answer_set(left: Iterable[str], right: Iterable[str]) -> bool:
    return {canonical_text(value) for value in left} == {
        canonical_text(value) for value in right
    }


def split_dataset(
    *,
    rog_rows: Iterable[dict[str, Any]],
    original_cwq_path: Path,
    nsm_test_path: Path,
    nsm_entities_path: Path,
    reverse_properties_path: Path,
    output_dir: Path,
    rog_source: dict[str, Any],
) -> dict[str, Any]:
    """Execute and atomically write the two complementary RoG-CWQ splits."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    original_by_id = load_original_cwq(original_cwq_path)
    nsm_by_id, needed_entity_indices = load_nsm_rows(nsm_test_path)
    entity_mid_by_index = load_needed_entity_mids(
        nsm_entities_path,
        needed_entity_indices,
    )
    reverse_properties = load_reverse_properties(reverse_properties_path)
    if original_by_id.keys() != nsm_by_id.keys():
        raise ValueError(
            "original CWQ and NSM test ID sets differ: "
            f"original_only={sorted(original_by_id.keys() - nsm_by_id.keys())[:10]}, "
            f"nsm_only={sorted(nsm_by_id.keys() - original_by_id.keys())[:10]}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )
    complete_path = temp_dir / "complete_witness.jsonl"
    no_witness_path = temp_dir / "no_witness.jsonl"
    details_path = temp_dir / "witness_details.jsonl"
    seen_ids: set[str] = set()
    counts = Counter()
    failure_reasons = Counter()

    try:
        with (
            complete_path.open("x", encoding="utf-8", newline="\n") as complete,
            no_witness_path.open("x", encoding="utf-8", newline="\n") as no_witness,
            details_path.open("x", encoding="utf-8", newline="\n") as details,
        ):
            for row_number, rog_row in enumerate(rog_rows, start=1):
                example_id = str(rog_row["id"]).strip()
                if example_id in seen_ids:
                    raise ValueError(f"duplicate RoG ID: {example_id}")
                seen_ids.add(example_id)

                original = original_by_id[example_id]
                nsm = nsm_by_id[example_id]
                rog_question = rog_row["question"]
                if not (
                    canonical_text(rog_question) == canonical_text(original["question"])
                    == canonical_text(nsm.question)
                ):
                    raise ValueError(f"question texts do not align: {example_id}")

                q_entities = validated_string_list(
                    rog_row.get("q_entity"),
                    field="q_entity",
                    example_id=example_id,
                )
                gold_answers = validated_string_list(
                    rog_row.get("a_entity"),
                    field="a_entity",
                    example_id=example_id,
                )
                if not same_answer_set(gold_answers, nsm.answers):
                    raise ValueError(f"RoG and NSM gold answers differ: {example_id}")
                graph = rog_row["graph"]

                topic_mids = [
                    entity_mid_by_index[index] for index in nsm.topic_indices
                ]
                mid_nodes = align_mid_constants(
                    original["sparql"],
                    topic_mids,
                    q_entities,
                )
                evaluation = evaluate_query(
                    parse_query(original["sparql"]),
                    graph,
                    mid_nodes,
                    gold_answers,
                    reverse_properties,
                )
                label = (
                    "complete_witness"
                    if evaluation["has_complete_witness"]
                    else "no_witness"
                )
                target = complete if label == "complete_witness" else no_witness
                target.write(
                    json.dumps(rog_row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                details.write(
                    json.dumps(
                        {
                            "row_number": row_number,
                            "id": example_id,
                            "classification": label,
                            "gold_answers": gold_answers,
                            "mid_bindings": mid_nodes,
                            **evaluation,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                counts[label] += 1
                if (
                    evaluation["has_complete_witness"]
                    and evaluation["inverse_witness_graph_indices"]
                ):
                    counts["complete_witness_using_inverse"] += 1
                if evaluation["failure_reason"]:
                    failure_reasons[evaluation["failure_reason"]] += 1
                if row_number % 100 == 0:
                    print(f"evaluated {row_number} RoG rows", flush=True)

        expected_ids = set(original_by_id)
        if seen_ids != expected_ids:
            raise ValueError(
                "RoG ID set differs from original CWQ/NSM: "
                f"missing={sorted(expected_ids - seen_ids)[:10]}, "
                f"extra={sorted(seen_ids - expected_ids)[:10]}"
            )
        manifest = {
            "version": 3,
            "criterion": (
                "complete_witness iff execution of the original CWQ gold SPARQL "
                "over the current directed RoG folded subgraph plus its "
                "schema-declared Freebase inverse-property closure returns at "
                "least one official RoG a_entity with positive graph evidence "
                "for every answer-changing condition; no_witness is its "
                "complement"
            ),
            "cvt_policy": (
                "A directed SPARQL path through removed CVT nodes matches one "
                "RoG predicate formed by joining the ordered atomic predicates with '-'"
            ),
            "inverse_property_policy": (
                "Only pairs declared by the supplied Freebase reverse_properties "
                "ontology are inverted. The inverse of folded r1-r2 is "
                "inverse(r2)-inverse(r1). Original graph indices are retained in "
                "witness details."
            ),
            "mid_binding_policy": (
                "NSM test.json entity indices are resolved through entities.txt and "
                "zipped in NSM/RoG topic order to the row's q_entity surfaces"
            ),
            "sources": {
                "rog": rog_source,
                "original_cwq": str(original_cwq_path.resolve()),
                "nsm_test": str(nsm_test_path.resolve()),
                "nsm_entities": str(nsm_entities_path.resolve()),
                "reverse_properties": {
                    "path": str(reverse_properties_path.resolve()),
                    "sha256": sha256_file(reverse_properties_path),
                    "official_source": GRAILQA_REVERSE_PROPERTIES_URL,
                    "relation_count": len(reverse_properties),
                },
            },
            "counts": {
                "total": len(seen_ids),
                "complete_witness": counts["complete_witness"],
                "no_witness": counts["no_witness"],
            },
            "complete_witness_using_inverse": counts[
                "complete_witness_using_inverse"
            ],
            "no_witness_reasons": dict(sorted(failure_reasons.items())),
            "outputs": {
                "complete_witness": "complete_witness.jsonl",
                "no_witness": "no_witness.jsonl",
                "details": "witness_details.jsonl",
            },
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_dir.replace(output_dir)
        return manifest
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def load_rog_rows(
    source: str,
    *,
    split: str,
    revision: str,
) -> tuple[Iterable[dict[str, Any]], dict[str, Any]]:
    local_path = Path(source)
    if local_path.is_file():
        return io.read_jsonl(local_path), {
            "kind": "jsonl",
            "path": str(local_path.resolve()),
        }

    from datasets import load_dataset
    from huggingface_hub import HfApi

    # Resolve the mutable revision first so the manifest identifies and the
    # loader reads the same commit.
    # Source: https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.dataset_info
    resolved_revision = HfApi().dataset_info(source, revision=revision).sha
    # Hugging Face documents streaming=True for incremental iteration without
    # downloading the complete dataset first.
    # Source: https://huggingface.co/docs/datasets/en/stream
    rows = load_dataset(
        source,
        split=split,
        revision=resolved_revision,
        streaming=True,
    )
    return rows, {
        "kind": "huggingface_dataset",
        "repo_id": source,
        "split": split,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct CWQ S/I splits by replaying official gold SPARQL "
            "on RoG input graphs."
        )
    )
    parser.add_argument(
        "--rog-source",
        default="rmanluo/RoG-cwq",
        help="Hugging Face dataset repo ID or a local RoG JSONL file",
    )
    parser.add_argument("--rog-split", default="test")
    parser.add_argument("--rog-revision", default="main")
    parser.add_argument("--original-cwq", type=Path, required=True)
    parser.add_argument("--nsm-test", type=Path, required=True)
    parser.add_argument("--nsm-entities", type=Path, required=True)
    parser.add_argument(
        "--reverse-properties",
        type=Path,
        required=True,
        help=(
            "GrailQA ontology/reverse_properties file used to close the RoG "
            "subgraph under declared Freebase inverse properties"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rog_rows, source = load_rog_rows(
        args.rog_source,
        split=args.rog_split,
        revision=args.rog_revision,
    )
    manifest = split_dataset(
        rog_rows=rog_rows,
        original_cwq_path=args.original_cwq,
        nsm_test_path=args.nsm_test,
        nsm_entities_path=args.nsm_entities,
        reverse_properties_path=args.reverse_properties,
        output_dir=args.output_dir,
        rog_source=source,
    )
    print(json.dumps({
        "total": manifest["counts"]["total"],
        "S": manifest["counts"]["complete_witness"],
        "I": manifest["counts"]["no_witness"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
