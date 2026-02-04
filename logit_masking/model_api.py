"""
Custom inspect_ai model provider with logit masking.

Registers an "hf-masked" ModelAPI that wraps a local HuggingFace model and
applies a MaskedReasoningProcessor during generation. Tokens inside
<reasoning>...</reasoning> tags are restricted to an allowed set determined
by the constraint's ``allowed_token_filter``.

This integrates with inspect_ai's standard generate() solver — no custom
solver is needed. The masking is transparent to the rest of the pipeline.

Usage (from code):
    from inspect_ai.model import get_model
    import logit_masking.model_api  # registers the provider

    model = get_model(
        "hf-masked/meta-llama/Llama-3.2-1B-Instruct",
        allowed_token_filter=emoji_token_filter,
    )
"""

import gc
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

import torch
from transformers import LogitsProcessorList

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatCompletionChoice,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from .processor import MaskedReasoningProcessor, load_model


def _messages_to_prompt(tokenizer, messages: list[ChatMessage]) -> str:
    """Convert inspect_ai ChatMessage list to a prompt string."""
    chat = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        chat.append({"role": msg.role, "content": content})

    try:
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for models without a chat template
        parts = []
        for msg in chat:
            role = msg["role"]
            if role == "system":
                parts.append(f"System: {msg['content']}")
            elif role == "user":
                parts.append(f"User: {msg['content']}")
            elif role == "assistant":
                parts.append(f"Assistant: {msg['content']}")
        parts.append("Assistant:")
        return "\n\n".join(parts)


@modelapi(name="hf-masked")
class MaskedHuggingFaceAPI(ModelAPI):
    """HuggingFace model provider with logit masking inside reasoning tags.

    Requires ``allowed_token_filter`` as a model arg — a callable that
    takes a tokenizer and returns the set of token IDs permitted inside
    ``<reasoning>`` tags.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=["HF_TOKEN"],
            config=config,
        )

        allowed_token_filter: Callable | None = model_args.get("allowed_token_filter")
        if allowed_token_filter is None:
            raise ValueError(
                "hf-masked provider requires 'allowed_token_filter' model arg: "
                "a callable(tokenizer) -> set[int]"
            )

        self.tokenizer, self.model = load_model(model_name)

        # Build the processor from the constraint's token filter
        allowed_ids = allowed_token_filter(self.tokenizer)
        allowed_ids.add(self.tokenizer.eos_token_id)

        end_ids = self.tokenizer.encode("</reasoning>", add_special_tokens=False)

        force_answer_prefix = model_args.get("force_answer_prefix")

        self.processor = MaskedReasoningProcessor(
            tokenizer=self.tokenizer,
            allowed_ids=allowed_ids,
            vocab_size=self.model.config.vocab_size,
            end_ids=end_ids,
            force_answer_prefix=force_answer_prefix,
        )

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        prompt = _messages_to_prompt(self.tokenizer, input)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        kwargs = {
            k: v
            for k, v in {
                "max_new_tokens": config.max_tokens or 512,
                "temperature": config.temperature,
                "do_sample": config.temperature is not None,
                "top_p": config.top_p,
                "top_k": config.top_k,
            }.items()
            if v is not None and v is not False
        }

        if config.stop_seqs:
            from transformers.generation import StopStringCriteria

            kwargs["stopping_criteria"] = [
                StopStringCriteria(self.tokenizer, config.stop_seqs)
            ]

        self.processor.reset()
        if config.max_tokens is not None:
            self.processor.max_masked_tokens = (
                config.max_tokens - MaskedReasoningProcessor.ANSWER_TOKEN_RESERVE
            )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **kwargs,
                logits_processor=LogitsProcessorList([self.processor]),
            )

        new_tokens = outputs[0][input_len:]
        last_token_id = new_tokens[-1].item()
        logger.info(
            f"Generation done: {len(new_tokens)} tokens, "
            f"last_token={last_token_id} ({self.tokenizer.decode([last_token_id])!r}), "
            f"is_eos={last_token_id == self.tokenizer.eos_token_id}, "
            f"mask_on={self.processor.mask_on}, "
            f"masked_count={self.processor._masked_count}"
        )
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(
                        content=response_text, source="generate"
                    ),
                    stop_reason="stop",
                )
            ],
            usage=ModelUsage(
                input_tokens=input_len,
                output_tokens=len(new_tokens),
                total_tokens=input_len + len(new_tokens),
            ),
        )

    def max_connections(self) -> int:
        # Processor has per-generation state; serialise calls.
        return 1

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
