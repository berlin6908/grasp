"""Answer prompts and response schema over cited evidence."""

from __future__ import annotations

import json


def response_schema() -> dict:
    return {
        "type": "object",
        "required": ["sufficient", "answers", "reason"],
        "additionalProperties": False,
        "properties": {
            "sufficient": {"type": "boolean"},
            "answers": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "reason": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "A concise answer justification that states the minimum concrete "
                    "facts, cites their evidence IDs, and derives every answer."
                ),
            },
        },
    }


def flatten_step_evidence(
    step_evidence: dict[str, dict],
) -> list[tuple[str, str, str]]:
    return [
        tuple(triple)
        for evidence in step_evidence.values()
        if "triples" in evidence
        for triple in evidence["triples"]
    ]


def render_flat_judge_prompt(
    question: str,
    step_evidence: dict[str, dict],
) -> str:
    triples = "\n".join(
        f"- [e{index}] " + json.dumps(
            triple,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for index, triple in enumerate(flatten_step_evidence(step_evidence), start=1)
    )
    return f"""# Role and task

You are the Final Reasoner in a knowledge-graph question-answering pipeline. Answer QUESTION from the supplied evidence. Only the directed triples establish facts.

# Evidence semantics

- `[eN] ["head","relation","tail"]` is one directed KG fact. Only the displayed relation and direction determine what it proves.
- Cite a fact in `reason` by stating it in natural language and placing its evidence ID immediately after it, for example: `France borders Germany [e1].`

# Output

Return one object matching the supplied schema. Set `sufficient` to true only when every fact needed by QUESTION is supported.

# Input

QUESTION
{question}

EVIDENCE
{triples}
"""


def render_judge_prompt(
    question: str,
    decision: str,
    step_evidence: dict[str, dict],
) -> str:
    steps = []
    for step_id, evidence in step_evidence.items():
        if evidence["op"] == "intersect":
            inputs = ", ".join(
                f"Step {input_step}" for input_step in evidence["inputs"]
            )
            steps.append(
                f"[{step_id}] INTERSECT: keep values shared by {inputs}."
            )
            continue

        triples = "\n".join(
            f"- [{step_id}.t{triple_index}] " + json.dumps(
                [head, relation, tail],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for triple_index, (head, relation, tail) in enumerate(
                evidence["triples"], start=1,
            )
        )
        evidence_block = triples or "- none selected"
        requirement = json.dumps(
            {
                "instruction": evidence["instruction"],
                "input": evidence["input"],
                "result_role": evidence["result_role"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        steps.append(
            f"[{step_id}] RETRIEVE\nREQUIREMENT\n{requirement}\n"
            f"EVIDENCE\n{evidence_block}"
        )
    rendered_evidence = "\n\n".join(steps)
    return f"""# Role and task

You are the Final Reasoner in a knowledge-graph question-answering pipeline. Answer QUESTION from the supplied Step Requirements and evidence. Verify that the evidence jointly satisfies every required Step and operation before answering. QUESTION defines the requested result. DECISION HINT describes how the Step results may be combined. Step Requirements and DECISION HINT are context; only the directed triples establish facts.

# Evidence semantics

- `[sN.tM] ["head","relation","tail"]` is one directed KG fact retrieved by Step `sN`. Only the displayed relation and direction determine what it proves.
- Cite a fact in `reason` by stating it in natural language and placing its evidence ID immediately after it, for example: `France borders Germany [s1.t1].`
- An Intersect Step keeps values shared by all listed input Steps.

# Output

Return one object matching the supplied schema. Set `sufficient` to true only when every fact and operation needed by QUESTION is supported.

- When sufficient, return every distinct supported answer. Copy entity and literal strings exactly; derive a scalar only from distinct supported values. In `reason`, give the shortest complete justification: state the minimum concrete facts needed for every answer, cite each fact with its evidence ID, and conclude with `Therefore, the answer is ...` or `Therefore, the answers are ...`. Include an intermediate inference only when a join, intersection, comparison, count, or other operation would otherwise be unclear.
- When insufficient, return an empty `answers` and use `reason` to identify the missing required fact or operation.

# Input

QUESTION
{question}

DECISION HINT
{decision}

STEP REQUIREMENTS AND EVIDENCE
{rendered_evidence}
"""
