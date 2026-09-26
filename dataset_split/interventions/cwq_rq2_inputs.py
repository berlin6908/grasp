"""System-specific graph freezing for the CWQ RQ2 interventions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from src.io import load_jsonl
from src.grounding.graph import Triple
from src.grounding.graph import normalize_graph


def frozen_ours_graph(
    graph: Iterable[Triple], deleted_triples: Iterable[Triple]
) -> list[Triple]:
    """Normalize G+ once, then apply the registered intervention."""

    normalized_graph = normalize_graph(graph)
    deleted = {tuple(triple) for triple in normalize_graph(deleted_triples)}
    return [triple for triple in normalized_graph if tuple(triple) not in deleted]


def preflight_ours_bundle(bundle: Path) -> dict:
    """Verify that stored Ours conditions are exactly the registered edits."""

    critical_path = bundle / "gminus.jsonl"
    if not critical_path.exists():
        critical_path = bundle / "gbinding.jsonl"
    if not critical_path.exists():
        raise ValueError(f"no critical graph file in {bundle}")

    rows_by_file = {
        "gplus": load_jsonl(bundle / "gplus.jsonl"),
        "critical": load_jsonl(critical_path),
        "sham": load_jsonl(bundle / "gsham.jsonl"),
        "intervention": load_jsonl(bundle / "interventions.jsonl"),
    }
    row_counts = {name: len(rows) for name, rows in rows_by_file.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"unaligned row counts: {row_counts}")

    details: list[dict] = []
    for gplus, critical, sham, intervention in zip(
        rows_by_file["gplus"],
        rows_by_file["critical"],
        rows_by_file["sham"],
        rows_by_file["intervention"],
        strict=True,
    ):
        example_id = intervention["id"]
        ids = {gplus["id"], critical["id"], sham["id"], example_id}
        if ids != {example_id}:
            raise ValueError(f"unaligned IDs around {example_id}: {sorted(ids)}")

        expected_critical = frozen_ours_graph(
            gplus["graph"], intervention["critical_triples"]
        )
        expected_sham = frozen_ours_graph(
            gplus["graph"], intervention["sham_triples"]
        )
        if normalize_graph(critical["graph"]) != expected_critical:
            raise ValueError(f"critical graph mismatch for {example_id}")
        if normalize_graph(sham["graph"]) != expected_sham:
            raise ValueError(f"sham graph mismatch for {example_id}")

        details.append(
            {
                "id": example_id,
                "status": "ok",
                "gplus_triples": len(normalize_graph(gplus["graph"])),
                "critical_visible_deleted": len(normalize_graph(gplus["graph"]))
                - len(expected_critical),
                "sham_visible_deleted": len(normalize_graph(gplus["graph"]))
                - len(expected_sham),
            }
        )

    return {
        "system": "ours",
        "bundle": str(bundle.resolve()),
        "rows": len(details),
        "critical_rows_valid": len(details),
        "sham_rows_valid": len(details),
        "critical_file": critical_path.name,
        "details": details,
    }


def _write_preflight(output_dir: Path, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    details = result["details"]
    with (output_dir / "preflight.jsonl").open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {key: value for key, value in result.items() if key != "details"}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check registered Grasp inputs for one CWQ RQ2 bundle."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    result = preflight_ours_bundle(args.bundle)
    if args.output_dir is not None:
        _write_preflight(args.output_dir, result)
    print(json.dumps({key: value for key, value in result.items() if key != "details"}))


if __name__ == "__main__":
    main()
