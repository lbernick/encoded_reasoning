"""
Logit masking for constrained reasoning.

Provides MaskedReasoningProcessor, a HuggingFace LogitsProcessor that
restricts which tokens the model can generate inside <reasoning>...</reasoning>
tags while leaving tokens outside those tags unconstrained.

This enables hard enforcement of token constraints (e.g. emoji-only reasoning)
at the decoding level, rather than relying on prompt-based soft enforcement.

Typical usage:
    tokenizer, model = load_model("meta-llama/Llama-3.2-1B-Instruct")
    allowed_ids = my_token_filter(tokenizer)

    start_ids = tokenizer.encode("<reasoning>", add_special_tokens=False)
    end_ids = tokenizer.encode("</reasoning>", add_special_tokens=False)

    processor = MaskedReasoningProcessor(allowed_ids, len(tokenizer), start_ids, end_ids)

    outputs = model.generate(
        **inputs,
        logits_processor=LogitsProcessorList([processor]),
    )
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor


def load_model(model_name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load a HuggingFace model and tokenizer.

    Attempts 8-bit quantisation first, falling back to fp16.

    Args:
        model_name: HuggingFace model name or path
            (e.g. "meta-llama/Llama-3.2-1B-Instruct").

    Returns:
        (tokenizer, model) tuple.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", load_in_8bit=True
        )
        print(f"Loaded {model_name} in 8-bit")
    except Exception as e:
        print(f"8-bit failed ({e}), falling back to fp16")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", dtype=torch.float16
        )
    return tokenizer, model


class MaskedReasoningProcessor(LogitsProcessor):
    """Restricts tokens generated inside <reasoning> tags to an allowed set.

    Tracks whether generation is currently inside a <reasoning>...</reasoning>
    block by watching for the start/end token sequences. While inside the block,
    only tokens in ``allowed_ids`` (plus the first token of the end sequence,
    so the model can close the block) are permitted — all others get -inf logits.

    When the model begins producing the end sequence (``</reasoning>``), the
    processor forces completion of that sequence token by token so it isn't
    left half-written.

    Args:
        allowed_ids: Set of token IDs permitted inside reasoning tags.
        vocab_size: Size of the model's vocabulary.
        start_ids: Token ID sequence for the opening tag (``<reasoning>``).
        end_ids: Token ID sequence for the closing tag (``</reasoning>``).
        max_masked_tokens: Maximum number of masked tokens before forcing the
            end sequence. After this many tokens are generated under masking,
            the processor forces ``</reasoning>`` so the model can produce a
            final answer. None disables the limit.
    """

    # Tokens reserved for </reasoning><answer>...</answer> after masking ends.
    ANSWER_TOKEN_RESERVE = 32

    def __init__(
        self,
        allowed_ids: set[int],
        vocab_size: int,
        start_ids: list[int],
        end_ids: list[int],
        max_masked_tokens: int | None = None,
    ):
        self.start_ids = start_ids
        self.end_ids = end_ids
        self.allowed_ids = set(allowed_ids)
        self.vocab_size = vocab_size
        self.max_masked_tokens = max_masked_tokens

        # Pre-build masks to avoid per-step allocation.
        # Normal masking: allowed content tokens + first token of end sequence.
        self.allowed_mask = self._build_mask(self.allowed_ids | {end_ids[0]})
        # Force masks: one per position in end_ids, to force sequential completion.
        self.force_masks = [
            self._build_mask({end_ids[i]}) for i in range(len(end_ids))
        ]
        self.mask_on = False
        self._masked_count = 0

    def reset(self):
        """Reset state between generation calls."""
        self.mask_on = False
        self._masked_count = 0

    def _build_mask(self, allowed_ids: set[int]) -> torch.Tensor:
        """Build an additive logit mask (0 for allowed, -inf for blocked)."""
        mask = torch.full((self.vocab_size,), float('-inf'))
        mask[list(allowed_ids)] = 0
        return mask

    def _end_seq_progress(self, input_ids: torch.Tensor) -> int:
        """Return how many trailing tokens match the start of end_ids."""
        tokens = input_ids[0].tolist()
        for n in range(min(len(self.end_ids), len(tokens)), 0, -1):
            if tokens[-n:] == self.end_ids[:n]:
                return n
        return 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # End sequence just completed → stop masking
        if input_ids.shape[-1] >= len(self.end_ids):
            if input_ids[0, -len(self.end_ids):].tolist() == self.end_ids:
                self.mask_on = False
                return scores

        # Start sequence just completed → start masking
        if not self.mask_on and input_ids.shape[-1] >= len(self.start_ids):
            if input_ids[0, -len(self.start_ids):].tolist() == self.start_ids:
                self.mask_on = True

        if not self.mask_on:
            return scores

        # Mid-way through end sequence → force completion
        progress = self._end_seq_progress(input_ids)
        if progress > 0:
            return scores + self.force_masks[progress].to(scores.device)

        self._masked_count += 1

        # Hit max masked tokens → force start of end sequence
        if self.max_masked_tokens is not None and self._masked_count >= self.max_masked_tokens:
            return scores + self.force_masks[0].to(scores.device)

        # Normal masking: allowed tokens + first token of end sequence
        return scores + self.allowed_mask.to(scores.device)
