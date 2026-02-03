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
    end_ids = tokenizer.encode("</reasoning>", add_special_tokens=False)

    processor = MaskedReasoningProcessor(tokenizer, allowed_ids, len(tokenizer), end_ids)

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

    Uses string-based tag detection to track whether generation is inside a
    reasoning block, robust to BPE merge differences across tokenizer contexts.

    Args:
        tokenizer: Tokenizer for decoding token IDs to text.
        allowed_ids: Token IDs permitted inside reasoning tags.
        vocab_size: Model vocabulary size.
        end_ids: Token IDs for the end tag, used for force-completion.
        max_masked_tokens: Force end sequence after this many masked tokens.
            None disables the limit.
        start_tag: Opening tag string.
        end_tag: Closing tag string.
    """

    ANSWER_TOKEN_RESERVE = 32

    def __init__(
        self,
        tokenizer,
        allowed_ids: set[int],
        vocab_size: int,
        end_ids: list[int],
        max_masked_tokens: int | None = None,
        start_tag: str = "<reasoning>",
        end_tag: str = "</reasoning>",
    ):
        self.tokenizer = tokenizer
        self.end_ids = end_ids
        self.allowed_ids = set(allowed_ids)
        self.vocab_size = vocab_size
        self.max_masked_tokens = max_masked_tokens
        self.start_tag = start_tag
        self.end_tag = end_tag

        # Allow any token whose decoded text is a prefix of end_tag
        # so the model can close the block regardless of BPE merges
        end_tag_token_ids = {
            tok_id for tok_id in range(vocab_size)
            if (decoded := tokenizer.decode([tok_id]))
            and self.end_tag.startswith(decoded)
        }

        self.allowed_mask = self._build_mask(self.allowed_ids | end_tag_token_ids)
        self.force_masks = [
            self._build_mask({end_ids[i]}) for i in range(len(end_ids))
        ]
        self.mask_on = False
        self._masked_count = 0
        self._forcing_end = False
        self._force_step = 0

    def reset(self):
        self.mask_on = False
        self._masked_count = 0
        self._forcing_end = False
        self._force_step = 0

    def _build_mask(self, allowed_ids: set[int]) -> torch.Tensor:
        mask = torch.full((self.vocab_size,), float('-inf'))
        mask[list(allowed_ids)] = 0
        return mask

    def _decoded_tail(self, input_ids: torch.Tensor, n_tokens: int = 10) -> str:
        return self.tokenizer.decode(input_ids[0, -n_tokens:].tolist())

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # Forcing end sequence token by token
        if self._forcing_end:
            step = self._force_step
            if step < len(self.end_ids):
                self._force_step += 1
                return scores + self.force_masks[step].to(scores.device)
            else:
                self._forcing_end = False
                self.mask_on = False
                return scores

        tail = self._decoded_tail(input_ids)

        # End tag completed → stop masking
        if self.mask_on and self.end_tag in tail:
            self.mask_on = False
            return scores

        # Start tag appeared → start masking
        if not self.mask_on and self.start_tag in tail:
            self.mask_on = True

        if not self.mask_on:
            return scores

        self._masked_count += 1

        # Hit max masked tokens → force end sequence
        if self.max_masked_tokens is not None and self._masked_count >= self.max_masked_tokens:
            self._forcing_end = True
            self._force_step = 1
            return scores + self.force_masks[0].to(scores.device)

        return scores + self.allowed_mask.to(scores.device)
