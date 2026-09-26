"""Specification-guided grounding by beam search with shared bindings."""

from __future__ import annotations

from ..planning.plan import IntersectStep, PlanDocument, RetrieveStep
from .graph import RelationIndex, adjacent_relations, expand_relation
from .state import (
    Case, GroundingCandidate, GroundedProof, ProgramAction, RankedAssignment,
    RelationDecision, coalesce_domain, copy_case, execute_intersect_rows,
    execute_union_rows, serialize_action, serialize_assignment,
)
from .signatures import RelationSignatureIndex


STATE_BEAM_SIZE = 8


class DagCraRetriever:
    def __init__(
        self,
        ranker,
        *,
        use_signatures: bool = True,
        enforce_intersections: bool = True,
    ):
        self.ranker = ranker
        self.use_signatures = use_signatures
        self.enforce_intersections = enforce_intersections

    def run(
        self,
        question: str,
        plan: PlanDocument,
        relation_index: RelationIndex,
        initial_rows: list[Case],
        ontology: RelationSignatureIndex,
    ) -> list[RankedAssignment]:
        states = [RankedAssignment((), 0.0, tuple(initial_rows))]
        cursor = 0
        while cursor < len(plan.steps):
            step = plan.steps[cursor]
            cursor += 1
            intersections: list[IntersectStep] = []
            while cursor < len(plan.steps) and isinstance(
                plan.steps[cursor], IntersectStep,
            ):
                intersections.append(plan.steps[cursor])
                cursor += 1
            states = self._retrieve(
                question,
                plan,
                step,
                tuple(intersections),
                relation_index,
                states,
                ontology,
            )
            if not states:
                break

        return states

    def _retrieve(
        self,
        question: str,
        plan: PlanDocument,
        step: RetrieveStep,
        intersections: tuple[IntersectStep, ...],
        relation_index: RelationIndex,
        states: list[RankedAssignment],
        ontology: RelationSignatureIndex,
    ) -> list[RankedAssignment]:
        expanded: list[RankedAssignment] = []
        for state in states:
            actions = enumerate_actions(step, state.rows, relation_index)
            executed: dict[ProgramAction, tuple[Case, ...]] = {}
            candidates: list[GroundingCandidate] = []
            for action in actions:
                signature = ontology.oriented_signature(
                    action.predicate, action.direction,
                )
                if self.use_signatures and signature is None:
                    continue
                source_type, target_type = (
                    signature if self.use_signatures else (None, None)
                )
                rows = tuple(execute_action(
                    step,
                    state.rows,
                    action,
                    relation_index,
                    source_type=source_type,
                    target_type=target_type,
                ))
                for intersection in intersections:
                    rows = tuple(
                        execute_intersect_rows(intersection, rows)
                        if self.enforce_intersections
                        else execute_union_rows(intersection, rows)
                    )
                if rows:
                    executed[action] = rows
                    candidates.append(GroundingCandidate(
                        action=action,
                        source_type=source_type,
                        target_type=target_type,
                        output_values=tuple(row.bindings[step.id] for row in rows),
                    ))
            input_values = tuple(sorted({
                row.bindings[step.input.key] for row in state.rows
            }))
            ranked = self.ranker.rank(
                question=question,
                plan=plan,
                step=step,
                assignment=state.assignment,
                input_values=input_values,
                candidates=tuple(candidates),
                limit=STATE_BEAM_SIZE,
            )
            expanded.extend(
                RankedAssignment(
                    assignment=(*state.assignment, (step.id, choice.action)),
                    score=state.score + choice.score,
                    rows=executed[choice.action],
                )
                for choice in ranked
            )
        return self._prune(expanded)

    @staticmethod
    def _prune(states: list[RankedAssignment]) -> list[RankedAssignment]:
        return sorted(
            states,
            key=lambda state: (-state.score, serialize_assignment(state.assignment)),
        )[:STATE_BEAM_SIZE]


def enumerate_actions(
    step: RetrieveStep,
    rows: tuple[Case, ...],
    relation_index: RelationIndex,
) -> tuple[ProgramAction, ...]:
    seeds = sorted({row.bindings[step.input.key] for row in rows})
    actions = (
        ProgramAction(predicate, direction)
        for predicate, direction in adjacent_relations(relation_index, seeds)
    )
    return tuple(sorted(actions, key=serialize_action))


def execute_action(
    step: RetrieveStep,
    rows: tuple[Case, ...],
    action: ProgramAction,
    relation_index: RelationIndex,
    *,
    source_type: str | None,
    target_type: str | None,
) -> list[Case]:
    completed: list[Case] = []
    for row in rows:
        seed = row.bindings[step.input.key]
        for value, triples in expand_relation(
            relation_index=relation_index,
            seed=seed,
            predicate=action.predicate,
            direction=action.direction,
        ):
            case = copy_case(row)
            case.bindings[step.id] = str(value).strip()
            evidence = tuple(
                tuple(str(part).strip() for part in triple)
                for triple in triples
            )
            decision = RelationDecision(
                step_id=step.id,
                instruction=step.instruction,
                result_role=step.result_role,
                seed=seed,
                predicate=action.predicate,
                direction=action.direction,
                source_type=source_type,
                target_type=target_type,
                representative_triple=list(triples[0]),
            )
            proof = GroundedProof(
                step_id=step.id,
                context_bindings=tuple(sorted(row.bindings.items())),
                output_value=case.bindings[step.id],
                evidence=evidence,
                relation_decision=decision,
                action=action,
            )
            case.proofs_by_step[step.id] = (proof,)
            completed.append(case)
    return coalesce_domain(completed)
