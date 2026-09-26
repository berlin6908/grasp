"""Build and load Freebase relation domain/range signatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from ..io import read_jsonl, sha256_file, write_json


INDEX_VERSION = 1


TypeFlow = tuple[str, str]


class RelationSignatureIndex:
    def __init__(self, signatures: dict[str, TypeFlow]) -> None:
        self.signatures = dict(signatures)

    def signature(self, relation: str) -> TypeFlow | None:
        return self.signatures.get(relation)

    def oriented_signature(
        self,
        relation: str,
        direction: str,
    ) -> TypeFlow | None:
        signature = self.signature(relation)
        if signature is None:
            return None
        if direction == "outgoing":
            return signature
        return signature[1], signature[0]

    def to_payload(self, *, sources: dict | None = None) -> dict:
        return {
            "version": INDEX_VERSION,
            "relations": {
                relation: list(signature)
                for relation, signature in sorted(self.signatures.items())
            },
            "sources": sources or {},
        }

    @classmethod
    def load(cls, path: Path) -> "RelationSignatureIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls({relation: tuple(types) for relation, types in payload["relations"].items()})


def build_ontology_index(
    fb_roles: Path,
    folded_relations: Iterable[str],
) -> RelationSignatureIndex:
    atomic = _parse_roles(fb_roles)
    signatures = dict(atomic)
    for relation in set(folded_relations):
        if relation in signatures:
            continue
        atoms = relation.split("-")
        if len(atoms) > 1 and all(atom in atomic for atom in atoms):
            signatures[relation] = (atomic[atoms[0]][0], atomic[atoms[-1]][1])
    return RelationSignatureIndex(signatures)


def _parse_roles(path: Path) -> dict[str, TypeFlow]:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()]
    clean = [tokens for tokens in rows if len(tokens) == 3]
    known_domains = {tokens[0] for tokens in clean}
    known_ranges = {tokens[2] for tokens in clean}
    records: list[tuple[str, str, str]] = []
    for tokens in rows:
        if len(tokens) == 3:
            records.append(tuple(tokens))
        elif len(tokens) == 5:
            left, right = _unique_split(tokens[2], known_ranges, known_domains)
            records.extend([
                (tokens[0], tokens[1], left),
                (right, tokens[3], tokens[4]),
            ])
        else:
            raise ValueError(f"Malformed fb_roles record in {path}: {tokens}")
    signatures: dict[str, TypeFlow] = {}
    for domain, relation, range_type in records:
        signature = (domain, range_type)
        previous = signatures.setdefault(relation, signature)
        if previous != signature:
            raise ValueError(f"Conflicting fb_roles signatures for {relation}")
    return signatures


def _unique_split(
    glued: str,
    known_left: set[str],
    known_right: set[str],
) -> TypeFlow:
    matches = [
        (glued[:index], glued[index:])
        for index in range(1, len(glued))
        if glued[:index] in known_left and glued[index:] in known_right
    ]
    if len(matches) != 1:
        raise ValueError(f"Glued ontology token has no unique boundary: {glued}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Freebase relation domain and range signatures for folded CWQ graphs."
    )
    parser.add_argument("--fb-roles", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, action="append", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def collect_relations(dataset_paths: list[Path]) -> set[str]:
    return {
        relation.strip()
        for path in dataset_paths
        for row in read_jsonl(path)
        for _, relation, _ in row["graph"]
    }


def main() -> None:
    args = parse_args()
    if args.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite ontology index: {args.output_path}")
    folded_relations = collect_relations(args.dataset_path)
    ontology = build_ontology_index(
        args.fb_roles,
        folded_relations,
    )
    sources = {
        "fb_roles": {
            "path": str(args.fb_roles.resolve()),
            "sha256": sha256_file(args.fb_roles),
        },
        "datasets": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.dataset_path
        ],
        "observed_relation_count": len(folded_relations),
    }
    write_json(args.output_path, ontology.to_payload(sources=sources))
    covered = sum(
        relation in ontology.signatures for relation in folded_relations
    )
    print(json.dumps({
        "output_path": str(args.output_path.resolve()),
        "atomic_and_macro_signature_count": len(ontology.signatures),
        "observed_relation_count": len(folded_relations),
        "covered_observed_relation_count": covered,
    }, indent=2))


if __name__ == "__main__":
    main()
