"""Messages and response schema for support specification induction."""

from __future__ import annotations

import json


PLANNER_SYSTEM_PROMPT = """# Role and goal

You plan evidence retrieval for a knowledge-graph question answering system. Given a question and linked entities, create the smallest complete Plan that retrieves the evidence a separate final reasoner needs. The Plan will be grounded and executed on the graph.

# Plan semantics

- Identify the requested answer role and the evidence needed to determine it.
- Each Retrieve asks one answer-relevant question-level semantic relation from its `input` to the values named by `result_role`. Runtime grounds it as one directed dataset relation; do not expose CVTs, mediator records, or storage paths as Steps.
- Start each branch from a linked entity. Chain Retrieves only when a distinct answer-changing relation needs an earlier Step's returned values as input; otherwise stop when the requested answer role is returned.
- Do not add a Retrieve merely to verify or describe values already returned by the answer relation. Retrieve a property separately only when the question asks for that value or the decision must compare, order, or aggregate it.
- Write each Retrieve `instruction` as a natural-language question containing `{input}` exactly once. Name its returned entities or values precisely in `result_role`.
- Use Intersect when independent branches return the same kind of entity or value and only their shared values satisfy the question.
- Write `decision` as a self-contained instruction for the final reasoner. Explain how to produce the answer from the Step results, including what to return and any condition, aggregation, or comparison to apply.

# Output

Return one JSON object that matches the supplied schema. Number the Steps consecutively from `s1`.

# Examples

### Direct answer relation

Input
{"linked_entities":[{"id":"a1","name":"Avery Chen"}],"question":"Where did Avery Chen go to school?"}

Output
{"steps":[{"id":"s1","op":"retrieve","instruction":"Which schools did {input} attend?","input":{"anchor":"a1"},"result_role":"school"}],"decision":"Return the schools from s1."}

### Relation composition

Input
{"linked_entities":[{"id":"a1","name":"Northwind Pictures"}],"question":"Who directed films produced by Northwind Pictures?"}

Output
{"steps":[{"id":"s1","op":"retrieve","instruction":"Which films were produced by {input}?","input":{"anchor":"a1"},"result_role":"film"},{"id":"s2","op":"retrieve","instruction":"Who directed {input}?","input":"s1","result_role":"director"}],"decision":"Return the directors from s2."}

### Intersection

Input
{"linked_entities":[{"id":"a1","name":"Mira Stone"},{"id":"a2","name":"Kai Lee"}],"question":"Which films were directed by Mira Stone and starred Kai Lee?"}

Output
{"steps":[{"id":"s1","op":"retrieve","instruction":"Which films were directed by {input}?","input":{"anchor":"a1"},"result_role":"film"},{"id":"s2","op":"retrieve","instruction":"Which films did {input} star in?","input":{"anchor":"a2"},"result_role":"film"},{"id":"s3","op":"intersect","inputs":["s1","s2"]}],"decision":"Return the films from s3."}

### Property condition

Input
{"linked_entities":[{"id":"a1","name":"Avery Chen"}],"question":"Which schools attended by Avery Chen were founded before 1900?"}

Output
{"steps":[{"id":"s1","op":"retrieve","instruction":"Which schools did {input} attend?","input":{"anchor":"a1"},"result_role":"school"},{"id":"s2","op":"retrieve","instruction":"When was {input} founded?","input":"s1","result_role":"founding date"}],"decision":"Return the schools from s1 whose founding dates in s2 are before 1900."}

### Comparison

Input
{"linked_entities":[{"id":"a1","name":"Film A"},{"id":"a2","name":"Film B"}],"question":"Which film came out first, Film A or Film B?"}

Output
{"steps":[{"id":"s1","op":"retrieve","instruction":"What was the release date of {input}?","input":{"anchor":"a1"},"result_role":"release date"},{"id":"s2","op":"retrieve","instruction":"What was the release date of {input}?","input":{"anchor":"a2"},"result_role":"release date"}],"decision":"Compare the release dates in s1 and s2. Return Film A if s1 is earlier; otherwise return Film B."}
"""


def build_planner_messages(
    question: str,
    contract: dict,
) -> list[dict[str, str]]:
    user_input = json.dumps(
        {
            "linked_entities": contract["linked_entities"],
            "question": question,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]


def _object_schema(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_TEXT_SCHEMA = {"type": "string", "minLength": 1}


_STEP_ID_SCHEMA = {**_TEXT_SCHEMA, "pattern": r"^s[1-9][0-9]*$"}


def build_planner_contract(
    linked_entities: list[str],
) -> dict:
    return {
        "linked_entities": [
            {"id": f"a{index}", "name": surface}
            for index, surface in enumerate(linked_entities, start=1)
        ],
    }


def planner_response_schema(contract: dict) -> dict:
    anchor_ids = [item["id"] for item in contract["linked_entities"]]
    anchor_schema = _object_schema({
        "anchor": {
            **_TEXT_SCHEMA,
            "enum": anchor_ids,
            "description": "One supplied linked-entity ID.",
        },
    })
    input_schema = {
        "anyOf": [
            anchor_schema,
            {**_STEP_ID_SCHEMA, "description": "Output from an earlier Step."},
        ],
    }
    return _object_schema({
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "anyOf": [
                    _object_schema({
                        "id": _STEP_ID_SCHEMA,
                        "op": {"const": "retrieve"},
                        "instruction": {
                            **_TEXT_SCHEMA,
                            "pattern": r"^.*\{input\}.*$",
                            "description": (
                                "A natural-language question asking for one answer-relevant "
                                "question-level semantic relation from the input to the "
                                "returned values."
                            ),
                        },
                        "input": input_schema,
                        "result_role": {
                            **_TEXT_SCHEMA,
                            "description": "A short noun phrase naming exactly the returned values.",
                        },
                    }),
                    _object_schema({
                        "id": _STEP_ID_SCHEMA,
                        "op": {"const": "intersect"},
                        "inputs": {
                            "type": "array",
                            "minItems": 2,
                            "uniqueItems": True,
                            "items": _STEP_ID_SCHEMA,
                        },
                    }),
                ],
            },
        },
        "decision": {
            **_TEXT_SCHEMA,
            "description": "How the final reasoner should produce the answer from the Step results.",
        },
    })
