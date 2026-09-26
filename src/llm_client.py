"""JSON client for frozen structured model calls."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from functools import lru_cache

from .contracts import FINAL_JUDGE_MODEL as DEFAULT_STRUCTURED_MODEL


class StructuredResponseError(ValueError):
    pass


@lru_cache(maxsize=4)
def _client(base_url: str):
    from openai import OpenAI

    return OpenAI(
        api_key="ollama",
        base_url=base_url,
        timeout=1000,
        max_retries=0,
    )


def generate(
    messages: list[dict[str, str]],
    *,
    json_schema: dict,
    usage_callback: Callable[[dict[str, int]], None] | None = None,
) -> str:
    response = _client(os.environ["OLLAMA_BASE_URL"]).chat.completions.create(
        model=DEFAULT_STRUCTURED_MODEL,
        messages=messages,
        temperature=0.0,
        seed=0,
        reasoning_effort="medium",
        max_tokens=4096,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": json_schema,
            },
        },
    )
    if usage_callback is not None:
        usage_callback({
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        })
    return response.choices[0].message.content or ""


def call_json(
    messages: list[dict[str, str]],
    *,
    json_schema: dict,
    usage_callback: Callable[[dict[str, int]], None] | None = None,
) -> dict:
    raw_output = generate(
        messages,
        json_schema=json_schema,
        usage_callback=usage_callback,
    )
    try:
        payload = json.loads(raw_output.strip())
    except json.JSONDecodeError as error:
        raise StructuredResponseError("Model response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise StructuredResponseError(
            f"Expected a JSON object from model {DEFAULT_STRUCTURED_MODEL}, got {type(payload).__name__}"
        )
    return payload
