"""
Integration test for logit-masked evaluation via the hf-masked model provider.

Runs a small eval (1-2 samples) through the full inspect pipeline with a local
HuggingFace model and an emoji constraint, verifying that logit masking is
active end-to-end.

Usage:
    python -m logit_masking.test_integration
    python -m logit_masking.test_integration --model meta-llama/Llama-3.2-1B-Instruct
    python -m logit_masking.test_integration --constraint only_emojis --dataset gsm8k -n 2
"""

import argparse

from evals.runner import run_eval, BASE_SYSTEM_PROMPT_COT, BASE_SYSTEM_PROMPT_NO_COT
from evals.constraints import CONSTRAINTS, get_constraint, register_constraint, ReasoningConstraint
from evals.datasets import DATASETS
from evals.token_filters import (
    logic_symbol_token_filter,
    number_token_filter,
    punctuation_token_filter,
    short_word_token_filter,
)

# Test-only constraints for exercising individual token filters
TEST_CONSTRAINTS = {
    "test_logic": ReasoningConstraint(
        name="test_logic",
        description="Reasoning with formal logic symbols only",
        system_prompt="Reason using only formal logic symbols.",
        allowed_token_filter=logic_symbol_token_filter,
    ),
    "test_numbers": ReasoningConstraint(
        name="test_numbers",
        description="Reasoning with numbers only",
        system_prompt="Reason using only numbers.",
        allowed_token_filter=number_token_filter,
    ),
    "test_punctuation": ReasoningConstraint(
        name="test_punctuation",
        description="Reasoning with punctuation only",
        system_prompt="Reason using only punctuation.",
        allowed_token_filter=punctuation_token_filter,
    ),
    "test_short_words": ReasoningConstraint(
        name="test_short_words",
        description="Reasoning with short words only (max 3 chars)",
        system_prompt="Reason using only very short words (3 characters or fewer).",
        allowed_token_filter=short_word_token_filter,
    ),
}

for constraint in TEST_CONSTRAINTS.values():
    register_constraint(constraint)

ALL_CONSTRAINTS = CONSTRAINTS


def main():
    parser = argparse.ArgumentParser(
        description="Integration test: logit-masked eval through full inspect pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name (without hf/ prefix — added automatically)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="gsm8k",
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "-c",
        "--constraint",
        default="test_short_words",
        choices=list(ALL_CONSTRAINTS.keys()),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--force-answer-prefix",
        type=bool,
        default=True,
        help="Force '\\n<answer>' after end tag",
    )
    args = parser.parse_args()

    model = f"hf/{args.model}"
    constraint = ALL_CONSTRAINTS[args.constraint]
    base_prompt = (
        BASE_SYSTEM_PROMPT_COT
        if constraint.expects_reasoning
        else BASE_SYSTEM_PROMPT_NO_COT
    )
    full_prompt = base_prompt + "\n" + constraint.system_prompt
    name = f"test_integration_{args.constraint}_{args.dataset}"

    print("Integration test — logit-masked eval")
    print(f"  Model:      {model}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Constraint: {args.constraint}")
    print(f"  Samples:    {args.n_samples}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"\n--- System Prompt ---\n{full_prompt}")
    print("--- End System Prompt ---\n")

    results = run_eval(
        constraint_name=args.constraint,
        model=model,
        dataset_name=args.dataset,
        n_samples=args.n_samples,
        name=name,
        max_tokens=args.max_tokens,
    )

    for log in results:
        print(f"Task: {log.eval.task}")
        print(f"Model: {log.eval.model}")
        if log.results:
            for metric in log.results.scores[0].metrics.values():
                print(f"  {metric.name}: {metric.value}")
        if log.samples:
            for sample in log.samples:
                print(f"\n--- Sample ---")
                print(f"Input:  {sample.input}")
                print(f"Target: {sample.target}")
                print(f"Output: {sample.output.completion}")


if __name__ == "__main__":
    main()
