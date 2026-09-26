"""Prompt for scoring a directed relation against support specification steps."""

from __future__ import annotations

import json

from ..planning.plan import PlanDocument, RetrieveStep, plan_to_payload
from .state import GroundingCandidate, ProgramAction, assignment_payload


OUTPUT_VALUE_CONTEXT_LIMIT = 32


def _render_relation(candidate: GroundingCandidate) -> str:
    action = candidate.action
    if candidate.source_type is None:
        if action.direction == "outgoing":
            return f"[INPUT] --{action.predicate}--> [OUTPUT]"
        return f"[OUTPUT] --{action.predicate}--> [INPUT]"
    if action.direction == "outgoing":
        return (
            f"[INPUT: {candidate.source_type}] --{action.predicate}--> "
            f"[OUTPUT: {candidate.target_type}]"
        )
    return (
        f"[OUTPUT: {candidate.target_type}] --{action.predicate}--> "
        f"[INPUT: {candidate.source_type}]"
    )


def _render_target(step: RetrieveStep, values: tuple[str, ...]) -> str:
    source = values[0] if len(values) == 1 else f"the current set of {len(values)} entities"
    return step.instruction.strip().replace("{input}", source)


def build_relation_prompt(
    question: str,
    plan: PlanDocument,
    step: RetrieveStep,
    assignment: tuple[tuple[str, ProgramAction], ...],
    input_values: tuple[str, ...],
    candidate: GroundingCandidate,
) -> str:
    visible_outputs = candidate.output_values[:OUTPUT_VALUE_CONTEXT_LIMIT]
    prior = assignment_payload(assignment)
    return "\n".join((
        "Judge the proposed relation for the current step.",
        (
            "Following it from the current input should produce the values requested "
            "by this step and fit the question and full plan."
        ),
        "Reply with exactly YES or NO.",
        f"Question: {question.strip()}",
        "Evidence DAG: " + json.dumps(
            {"steps": plan_to_payload(plan)["steps"]},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "Relations chosen for earlier steps: " + json.dumps(
            prior, ensure_ascii=False, separators=(",", ":"),
        ),
        f"Current step asks for: {_render_target(step, input_values)}",
        f"Current result role: {step.result_role}",
        f"Proposed directed relation: {_render_relation(candidate)}",
        "Deterministic output values shown: " + json.dumps(
            visible_outputs, ensure_ascii=False, separators=(",", ":"),
        ),
        f"Actual output count: {len(candidate.output_values)}",
        "Answer:",
    ))
