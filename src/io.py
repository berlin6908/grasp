"""Shared file operations for datasets and run artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any, Literal


def load_jsonl(path: Path) -> list[Any]:
    """Load the current JSONL record format."""
    return list(read_jsonl(path))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream one JSON object per line."""
    with path.open(encoding="utf-8") as handle:
        yield from map(json.loads, handle)


def iter_dataset_samples(
    path: Path,
    *,
    start_example: int | None = None,
    end_example: int | None = None,
) -> Iterator[tuple[int, dict]]:
    """Select rows while retaining their one-based source indices."""
    start = 0 if start_example is None else start_example - 1
    yield from islice(enumerate(read_jsonl(path), start=1), start, end_example)


def write_jsonl(
    path: Path, rows: Iterable[dict], *, mode: Literal["w", "x"] = "w",
) -> None:
    """Write records in order; mode='x' refuses to replace an existing artifact."""

    with path.open(mode, encoding="utf-8", buffering=1) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON artifact after creating its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def index_rows(rows: Iterable[dict], id_field: str, label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        example_id = str(row[id_field])
        if example_id in indexed:
            raise ValueError(f"duplicate {label} id: {example_id}")
        indexed[example_id] = row
    return indexed


def require_aligned_ids(expected: set[str], observed: set[str], label: str) -> None:
    if expected == observed:
        return
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    raise ValueError(
        f"{label} ids do not match the registered population: "
        f"missing={missing[:3]} ({len(missing)}), extra={extra[:3]} ({len(extra)})"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def source_snapshot_sha256() -> str:
    root = Path(__file__).resolve().parent
    return sha256_tree(root, list(root.rglob("*.py")))
