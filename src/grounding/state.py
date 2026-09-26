"""Grounding states, binding operations, and deterministic action encoding."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Iterable

from ..planning.plan import IntersectStep
from .graph import Triple, normalize_value


Bindings = dict[str, str]


@dataclass(frozen=True)
class RelationDecision:
    step_id: str
    instruction: str
    result_role: str
    seed: str
    predicate: str
    direction: str
    source_type: str | None
    target_type: str | None
    representative_triple: Triple

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "instruction": self.instruction,
            "result_role": self.result_role,
            "seed": self.seed,
            "predicate": self.predicate,
            "direction": self.direction,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "representative_triple": list(self.representative_triple),
        }


@dataclass(frozen=True)
class ProgramAction:
    predicate: str
    direction: str


@dataclass(frozen=True)
class GroundingCandidate:
    """One executable action with binding-conditioned ranking evidence."""

    action: ProgramAction
    source_type: str | None
    target_type: str | None
    output_values: tuple[str, ...]

    def __post_init__(self) -> None:
        source_type = (
            str(self.source_type).strip() if self.source_type is not None else None
        )
        target_type = (
            str(self.target_type).strip() if self.target_type is not None else None
        )
        output_values = tuple(sorted(
            {
                str(value).strip()
                for value in self.output_values
                if str(value).strip()
            },
            key=lambda value: (normalize_value(value), value),
        ))
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "target_type", target_type)
        object.__setattr__(self, "output_values", output_values)


@dataclass(frozen=True)
class GroundedProof:
    """One directed runtime relation executed for one Retrieve Step."""

    step_id: str
    context_bindings: tuple[tuple[str, str], ...]
    output_value: str
    evidence: tuple[tuple[str, str, str], ...]
    relation_decision: RelationDecision
    action: ProgramAction


@dataclass
class Case:
    """One explicit binding row with factorized terminal proofs by Retrieve Step."""

    bindings: Bindings = field(default_factory=dict)
    proofs_by_step: dict[str, tuple[GroundedProof, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredAction:
    action: ProgramAction
    score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("Relation score must be finite")


CanonicalAssignment = tuple[tuple[str, ProgramAction], ...]


@dataclass(frozen=True)
class RankedAssignment:
    assignment: CanonicalAssignment
    score: float
    rows: tuple[Case, ...]


def action_payload(action: ProgramAction) -> dict[str, str]:
    return {"predicate": action.predicate, "direction": action.direction}


def assignment_payload(assignment: CanonicalAssignment) -> dict[str, dict[str, str]]:
    return {step_id: action_payload(action) for step_id, action in assignment}


def serialize_action(action: ProgramAction) -> str:
    return json.dumps(
        action_payload(action),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def serialize_assignment(assignment: CanonicalAssignment) -> str:
    return json.dumps(
        assignment_payload(assignment),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _proof_identity(proof: GroundedProof) -> tuple:
    decision = proof.relation_decision
    return (
        proof.step_id,
        tuple(
            (name, normalize_value(value))
            for name, value in proof.context_bindings
        ),
        normalize_value(proof.output_value),
        (proof.action.predicate, proof.action.direction),
        proof.evidence,
        (
            normalize_value(decision.seed),
            decision.predicate,
            decision.direction,
            tuple(decision.representative_triple),
        ),
    )


def _proof_order(proof: GroundedProof) -> tuple:
    decision = proof.relation_decision
    return (
        (proof.action.predicate, proof.action.direction),
        len(proof.evidence),
        proof.evidence,
        (decision.predicate, decision.direction),
    )


def _merge_proof_groups(
    *groups: dict[str, tuple[GroundedProof, ...]],
) -> dict[str, tuple[GroundedProof, ...]]:
    by_step: dict[str, dict[tuple, GroundedProof]] = {}
    for group in groups:
        for step_id, proofs in group.items():
            step_proofs = by_step.setdefault(step_id, {})
            for proof in proofs:
                step_proofs.setdefault(_proof_identity(proof), proof)
    return {
        step_id: tuple(sorted(proofs.values(), key=_proof_order))
        for step_id, proofs in sorted(by_step.items())
    }


def _binding_key(case: Case) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((name, normalize_value(value)) for name, value in case.bindings.items()))


def coalesce_domain(cases: Iterable[Case]) -> list[Case]:
    """Keep one binding row and the union of its atomic proof factors."""

    merged: dict[tuple[tuple[str, str], ...], Case] = {}
    for case in cases:
        key = _binding_key(case)
        current = merged.get(key)
        if current is None:
            merged[key] = Case(
                bindings=dict(case.bindings),
                proofs_by_step=_merge_proof_groups(case.proofs_by_step),
            )
        else:
            current.proofs_by_step = _merge_proof_groups(
                current.proofs_by_step, case.proofs_by_step,
            )
    return list(merged.values())


def execute_intersect_rows(step: IntersectStep, rows: tuple[Case, ...]) -> list[Case]:
    output: list[Case] = []
    for row in rows:
        values = [row.bindings[ref.key] for ref in step.inputs]
        if len({normalize_value(value) for value in values}) != 1:
            continue
        case = copy_case(row)
        case.bindings[step.id] = values[0]
        output.append(case)
    return coalesce_domain(output)


def execute_union_rows(step: IntersectStep, rows: tuple[Case, ...]) -> list[Case]:
    output: list[Case] = []
    for row in rows:
        values: dict[str, str] = {}
        for ref in step.inputs:
            value = row.bindings[ref.key]
            key = normalize_value(value)
            current = values.get(key)
            if current is None or value < current:
                values[key] = value
        for key in sorted(values):
            case = copy_case(row)
            case.bindings[step.id] = values[key]
            output.append(case)
    return coalesce_domain(output)


def copy_case(case: Case) -> Case:
    return Case(
        bindings=dict(case.bindings),
        proofs_by_step=dict(case.proofs_by_step),
    )
