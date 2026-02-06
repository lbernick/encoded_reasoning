"""
Smoke test for logit-masked generation.

Loads a model, builds an emoji token filter, and generates a short
response with masking active inside <reasoning> tags.

Usage:
    python -m logit_masking.test_masking
    python -m logit_masking.test_masking --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse

import torch
from transformers import LogitsProcessorList

from .processor import MaskedReasoningProcessor, load_model
from ..evals.token_filters import emoji_token_filter


def main():
    parser = argparse.ArgumentParser(description="Test logit-masked emoji generation")
    parser.add_argument(
        "--model",
        default="openai-community/gpt2",
        help="HuggingFace model name to load",
    )
    parser.add_argument(
        "--prompt",
        default="Express happiness in emojis only:",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)

    # Build allowed set
    allowed_ids = emoji_token_filter(tokenizer)
    allowed_ids.add(tokenizer.eos_token_id)

    end_ids = tokenizer.encode("</reasoning>", add_special_tokens=False)

    processor = MaskedReasoningProcessor(
        tokenizer, allowed_ids, model.config.vocab_size, end_ids
    )

    # Tokenize prompt and append <reasoning> prefix
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    reasoning_prefix = tokenizer.encode(
        "<reasoning>", add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], reasoning_prefix], dim=-1)
    inputs["attention_mask"] = torch.cat(
        [inputs["attention_mask"], torch.ones_like(reasoning_prefix)], dim=-1
    )

    processor.reset()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=5,
            logits_processor=LogitsProcessorList([processor]),
        )

    print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    main()
