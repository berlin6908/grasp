"""Frozen-Qwen scoring of executable graph relations."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..planning.plan import PlanDocument, RetrieveStep
from .state import GroundingCandidate, ProgramAction, ScoredAction, serialize_action
from .prompt import OUTPUT_VALUE_CONTEXT_LIMIT, build_relation_prompt


class ActionContextOverflowError(RuntimeError):
    pass


class QwenRelationRanker:
    """Score every viable relation with one frozen YES/NO decision."""

    def __init__(
        self,
        model_name: str,
        model_revision: str,
        scoring_batch_size: int,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.scoring_batch_size = scoring_batch_size
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
            local_files_only=True,
            use_fast=True,
            padding_side="left",
        )
        self.yes_token_id, = self.tokenizer.encode("YES", add_special_tokens=False)
        self.no_token_id, = self.tokenizer.encode("NO", add_special_tokens=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_revision,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()

    def rank(
        self,
        *,
        question: str,
        plan: PlanDocument,
        step: RetrieveStep,
        assignment: tuple[tuple[str, ProgramAction], ...],
        input_values: tuple[str, ...],
        candidates: tuple[GroundingCandidate, ...],
        limit: int,
    ) -> list[ScoredAction]:
        if not candidates:
            return []
        prompts = [
            build_relation_prompt(
                question,
                plan,
                step,
                assignment,
                input_values,
                candidate,
            )
            for candidate in candidates
        ]
        scores = self._score_prompts(prompts)
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-item[1], serialize_action(item[0].action)),
        )
        return [
            ScoredAction(candidate.action, score)
            for candidate, score in ranked[:limit]
        ]

    def _score_prompts(self, prompts: list[str]) -> list[float]:
        prompt_ids = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            for prompt in prompts
        ]
        context_limit = min(
            self.model.config.max_position_embeddings,
            self.tokenizer.model_max_length,
        )
        if any(len(ids) + 1 > context_limit for ids in prompt_ids):
            raise ActionContextOverflowError(
                "Whole-Plan relation scoring exceeds the Qwen context"
            )

        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(prompt_ids), self.scoring_batch_size):
                batch = prompt_ids[start:start + self.scoring_batch_size]
                encoded = self.tokenizer.pad(
                    {"input_ids": batch}, padding=True,
                    return_attention_mask=True, return_tensors="pt",
                ).to(self.device)
                attention_mask = encoded["attention_mask"]
                position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
                logits = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[:, -1].float()
                margins = logits[:, self.yes_token_id] - logits[:, self.no_token_id]
                scores.extend(torch.nn.functional.logsigmoid(margins).tolist())
        return scores

    def runtime_metadata(self) -> dict:
        return {
            "qwen_model": self.model_name,
            "qwen_model_configured_revision": self.model_revision,
            "qwen_model_loaded_revision": self.model.config._commit_hash,
            "scoring_batch_size": self.scoring_batch_size,
            "output_value_context_limit": OUTPUT_VALUE_CONTEXT_LIMIT,
        }
