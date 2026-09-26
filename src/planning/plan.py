"""Reasoning-plan data model and JSON encoding."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StepRef:
    step_id: str

    def to_raw(self) -> str:
        return self.step_id

    @property
    def key(self) -> str:
        return self.step_id


@dataclass(frozen=True)
class AnchorRef:
    """A model-visible retrieval starting point derived from the question."""

    surface: str

    def __post_init__(self) -> None:
        surface = str(self.surface).strip()
        object.__setattr__(self, "surface", surface)

    @property
    def key(self) -> str:
        return f"anchor:{self.surface}"

    def to_raw(self) -> dict[str, str]:
        return {"anchor": self.surface}


@dataclass(frozen=True)
class RetrieveStep:
    id: str
    instruction: str
    input: StepRef | AnchorRef
    result_role: str
    op: str = field(init=False, default="retrieve")

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_role", self.result_role.strip())


@dataclass(frozen=True)
class IntersectStep:
    id: str
    inputs: tuple[StepRef, ...]
    op: str = field(init=False, default="intersect")


PlanStep = RetrieveStep | IntersectStep


def step_refs(step: PlanStep) -> tuple[StepRef | AnchorRef, ...]:
    if isinstance(step, RetrieveStep):
        return (step.input,)
    return step.inputs


@dataclass(frozen=True)
class PlanDocument:
    """One role-annotated evidence-acquisition DAG plus a Final decision."""

    steps: tuple[PlanStep, ...]
    decision: str

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("Plan steps must not be empty")
        object.__setattr__(self, "decision", self.decision.strip())
        self._validate_structure()

    def _validate_structure(self) -> None:
        declared: set[str] = set()

        for index, step in enumerate(self.steps, start=1):
            expected_id = f"s{index}"
            if step.id != expected_id:
                raise ValueError(f"Plan step ids must be consecutive: expected {expected_id}, got {step.id}")
            refs = step_refs(step)
            if isinstance(step, RetrieveStep):
                if step.instruction.count("{input}") != 1:
                    raise ValueError(
                        f"Retrieve step {step.id} instruction must contain {{input}} exactly once"
                    )
            if isinstance(step, IntersectStep) and len(step.inputs) < 2:
                raise ValueError(f"Intersect step {step.id} requires at least two inputs")
            if isinstance(step, IntersectStep) and len(set(step.inputs)) != len(step.inputs):
                raise ValueError(f"Intersect step {step.id} requires distinct inputs")
            for ref in refs:
                if isinstance(ref, StepRef) and ref.step_id not in declared:
                    raise ValueError(
                        f"Reference {ref.step_id!r} in {step.id} must target a declared "
                        f"earlier sN Step; available refs: {sorted(declared)}"
                    )
            declared.add(step.id)


def _string(raw: object, *, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return raw.strip()


def _input(
    raw: object, *, label: str, anchors: dict[str, str] | None,
) -> StepRef | AnchorRef:
    if isinstance(raw, str):
        return StepRef(_string(raw, label=label))
    if isinstance(raw, dict) and set(raw) == {"anchor"}:
        anchor = _string(raw["anchor"], label=f"{label} anchor")
        if anchors is not None:
            if anchor not in anchors:
                raise ValueError(f"{label} uses unknown anchor ID: {anchor}")
            anchor = _string(anchors[anchor], label=f"{label} anchor")
        return AnchorRef(anchor)
    raise ValueError(f"{label} must be an Anchor or earlier sN Step")


def _parse_step(raw: dict, *, index: int, anchors: dict[str, str] | None):
    step_id = _string(raw["id"], label=f"Plan step {index} id")
    op = raw["op"]
    label = f"Plan step {step_id}"
    if op == "retrieve":
        return RetrieveStep(
            step_id,
            _string(raw["instruction"], label=f"{label} instruction"),
            _input(raw["input"], label=f"{label} input", anchors=anchors),
            _string(raw["result_role"], label=f"{label} result_role"),
        )
    if op == "intersect":
        inputs = raw["inputs"]
        return IntersectStep(
            step_id,
            tuple(StepRef(_string(item, label=f"{label} input")) for item in inputs),
        )
    raise ValueError(f"{label} has unsupported op: {op}")


def plan_from_payload(
    payload: dict, *, anchors: dict[str, str] | None = None,
) -> PlanDocument:
    return PlanDocument(
        tuple(
            _parse_step(step, index=index, anchors=anchors)
            for index, step in enumerate(payload["steps"], start=1)
        ),
        _string(payload["decision"], label="Plan decision"),
    )


def plan_from_planner_payload(payload: dict, contract: dict) -> PlanDocument:
    return plan_from_payload(
        payload,
        anchors={item["id"]: item["name"] for item in contract["linked_entities"]},
    )


def plan_to_payload(plan: PlanDocument) -> dict:
    steps = []
    for step in plan.steps:
        if isinstance(step, RetrieveStep):
            steps.append({
                "id": step.id,
                "op": step.op,
                "instruction": step.instruction,
                "input": step.input.to_raw(),
                "result_role": step.result_role,
            })
        else:
            steps.append({
                "id": step.id,
                "op": step.op,
                "inputs": [ref.to_raw() for ref in step.inputs],
            })
    return {
        "steps": steps,
        "decision": plan.decision,
    }
