"""
CLI entry point for the constrained reasoning evaluation framework.

Usage:
    python -m evals --model openrouter/openai/gpt-4o-mini --dataset gsm8k -n 10
    python -m evals -m openrouter/openai/gpt-4o-mini -d gsm8k -c no_cot -n 50
    python -m evals -m openrouter/anthropic/claude-3.5-sonnet -d gsm8k -n 50 --seed 123
    python -m evals --name "baseline_gpt4o" -c baseline -n 100
"""

import argparse

from .runner import run_eval
from .datasets import DATASETS
from .constraints import CONSTRAINTS


def short_model_name(model: str) -> str:
    """Extract short model name from full path (e.g., 'openrouter/openai/gpt-4o-mini' -> 'gpt-4o-mini')."""
    return model.split("/")[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Run constrained reasoning evaluations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m", "--model",
        default="openrouter/openai/gpt-4o-mini",
        help="Model to evaluate. Use 'hf/model-name' for local HuggingFace models "
             "(enables logit masking when constraint supports it), or OpenRouter format for API models.",
    )
    parser.add_argument(
        "-d", "--dataset",
        default="gsm8k",
        choices=list(DATASETS.keys()),
        help="Dataset to use",
    )
    parser.add_argument(
        "-c", "--constraint",
        required=True,
        choices=list(CONSTRAINTS.keys()),
        help="Reasoning constraint to apply",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name for this eval run (shows in logs). Defaults to '{constraint}_{dataset}'",
    )
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=10,
        help="Number of samples from the dataset to evaluate (0 = full dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of times to run each sample (reduces variance via majority vote)",
    )
    parser.add_argument(
        "--repeat-input",
        type=int,
        default=1,
        help="Number of times to repeat the question in the prompt",
    )
    parser.add_argument(
        "--filler-tokens",
        type=int,
        default=0,
        help="Number of filler tokens (periods) to add to the prompt",
    )

    args = parser.parse_args()

    # Build eval name
    eval_name = args.name or f"{args.constraint}_{args.dataset}_{short_model_name(args.model)}"
    if args.repeat_input > 1:
        eval_name = f"{eval_name}_repeat{args.repeat_input}"

    print("Running evaluation...")
    print(f"  Name:       {eval_name}")
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Constraint: {args.constraint}")
    print(f"  Samples:    {args.n_samples}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Repeat:     {args.repeat_input}")
    print(f"  Filler Tokens: {args.filler_tokens}")
    print(f"  Seed:       {args.seed}")
    print()

    results = run_eval(
        constraint_name=args.constraint,
        model=args.model,
        dataset_name=args.dataset,
        n_samples=args.n_samples if args.n_samples > 0 else None,
        seed=args.seed,
        epochs=args.epochs,
        repeat_input=args.repeat_input,
        filler_tokens=args.filler_tokens,
        name=eval_name,
    )

    return results


if __name__ == "__main__":
    main()
