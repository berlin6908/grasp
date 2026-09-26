"""Connect a saved support specification to specification-guided grounding and evidence records."""

from __future__ import annotations

import json
from time import perf_counter

from .contracts import RESULT_VERSION
from .planning.plan import AnchorRef, PlanDocument, RetrieveStep, plan_to_payload
from .grounding.graph import RelationIndex, build_relation_index, normalize_graph
from .grounding.state import Case, GroundedProof, action_payload, assignment_payload
from .grounding.scorer import ActionContextOverflowError
from .grounding.signatures import RelationSignatureIndex


class Pipeline:
    def __init__(self, planner, retriever, ontology: RelationSignatureIndex):
        self.planner = planner
        self.retriever = retriever
        self.ontology = ontology

    def run(
        self,
        question: str,
        graph: list[list[str]],
        q_entities: list[str] | tuple[str, ...],
        *,
        example_id: str,
    ) -> dict:
        started = perf_counter()
        full_graph = normalize_graph(graph)
        full_index = build_relation_index(full_graph)
        try:
            plan = self.planner.run(example_id, question)
        except ValueError:
            result = self._result("error", error_type="plan_invalid")
        else:
            result = self._run_plan(
                question,
                plan,
                full_graph,
                full_index,
                q_entities,
            )
        result["timings"] = {"total_seconds": perf_counter() - started}
        return result

    def _run_plan(
        self,
        question: str,
        plan: PlanDocument,
        full_graph: list[list[str]],
        full_index: RelationIndex,
        q_entities: list[str] | tuple[str, ...],
    ) -> dict:
        semantic_program = plan_to_payload(plan)
        rows, anchor_error = self._anchor_rows(plan, full_graph, q_entities)
        if anchor_error:
            return self._result(
                "insufficient",
                error_type=anchor_error,
                semantic_program=semantic_program,
            )

        try:
            ranked = self.retriever.run(
                question, plan, full_index, rows, self.ontology,
            )
        except ActionContextOverflowError:
            return self._result(
                "error",
                error_type="action_context_overflow",
                semantic_program=semantic_program,
            )
        if not ranked:
            return self._result(
                "insufficient",
                error_type="retrieval_failed",
                semantic_program=semantic_program,
            )

        ranked_rows = [
            (rank, result, row)
            for rank, result in enumerate(ranked, start=1)
            for row in result.rows
        ]
        rows = [row for _rank, _result, row in ranked_rows]
        proof_bank, proof_ids = self._intern_proofs(rows)
        symbolic = [
            {
                "row_id": index,
                "assignment_rank": rank,
                "assignment_score": assignment.score,
                "assignment": assignment_payload(assignment.assignment),
                "bindings": dict(sorted(row.bindings.items())),
                "proof_refs_by_step": {
                    step_id: [
                        proof_ids[self._proof_serialization_key(proof)]
                        for proof in proofs
                    ]
                    for step_id, proofs in sorted(row.proofs_by_step.items())
                },
            }
            for index, (rank, assignment, row) in enumerate(ranked_rows, start=1)
        ]
        return self._result(
            "ok",
            semantic_program=semantic_program,
            proof_bank=proof_bank,
            evidence_rows=symbolic,
            assignments=self._assignments_payload(ranked),
        )

    @staticmethod
    def _anchor_rows(
        plan: PlanDocument,
        graph: list[list[str]],
        q_entities: list[str] | tuple[str, ...],
    ):
        nodes = {entity for triple in graph for entity in (triple[0], triple[2])}
        anchors = dict.fromkeys([
            step.input
            for step in plan.steps
            if isinstance(step, RetrieveStep) and isinstance(step.input, AnchorRef)
        ])
        bindings = {}
        for ref in anchors:
            if ref.surface not in q_entities or ref.surface not in nodes:
                return [], "unresolved_anchor"
            bindings[ref.key] = ref.surface
        return [Case(bindings)], None

    @staticmethod
    def _serialize_proof(proof: GroundedProof) -> dict:
        return {
            "step_id": proof.step_id,
            "context_bindings": dict(proof.context_bindings),
            "output_value": proof.output_value,
            "proof": [list(triple) for triple in proof.evidence],
            "relation_decision": proof.relation_decision.to_dict(),
            "action": action_payload(proof.action),
        }

    @staticmethod
    def _assignments_payload(ranked) -> list[dict]:
        return [
            {
                "rank": rank,
                "score": item.score,
                "actions": assignment_payload(item.assignment),
                "row_count": len(item.rows),
            }
            for rank, item in enumerate(ranked, start=1)
        ]

    @classmethod
    def _proof_serialization_key(cls, proof: GroundedProof) -> str:
        return json.dumps(
            cls._serialize_proof(proof),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def _intern_proofs(cls, rows: list[Case]) -> tuple[list[dict], dict[str, str]]:
        serialized = {
            cls._proof_serialization_key(proof): cls._serialize_proof(proof)
            for row in rows
            for proofs in row.proofs_by_step.values()
            for proof in proofs
        }
        proof_ids = {
            key: f"p{index:06d}"
            for index, key in enumerate(sorted(serialized), start=1)
        }
        return [
            {"proof_id": proof_ids[key], **serialized[key]}
            for key in sorted(serialized)
        ], proof_ids

    @staticmethod
    def _result(
        status: str,
        *,
        error_type: str | None = None,
        semantic_program: dict | None = None,
        proof_bank: list[dict] | None = None,
        evidence_rows: list[dict] | None = None,
        assignments: list[dict] | None = None,
    ) -> dict:
        return {
            "result_version": RESULT_VERSION,
            "status": status,
            "error_type": error_type,
            "semantic_program": semantic_program or {},
            "proof_bank": proof_bank or [],
            "evidence_rows": evidence_rows or [],
            "assignments": assignments or [],
        }
