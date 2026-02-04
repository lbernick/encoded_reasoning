"""
CLI entry point for the constrained reasoning evaluation framework.

Usage:
    # Single-stage evaluations
    python -m evals --model openrouter/openai/gpt-4o-mini --dataset gsm8k -n 10
    python -m evals -m openrouter/openai/gpt-4o-mini -d gsm8k -c no_cot -n 50
    python -m evals -m openrouter/anthropic/claude-3.5-sonnet -d gsm8k -n 50 --seed 123
    python -m evals --name "baseline_gpt4o" -c baseline -n 100

    # Two-stage evaluations (reason first, then answer)
    python -m evals --two-stage -d gsm8k -n 10
    python -m evals --two-stage -c encoded -d gsm8k -n 50
"""

import argparse
import os

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
        "-m",
        "--model",
        default=os.environ.get("MODEL", "openrouter/openai/gpt-4o-mini"),
        help="Model to evaluate. Use 'hf/model-name' for local HuggingFace models "
        "(enables logit masking when constraint supports it), or OpenRouter format for API models.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=os.environ.get("DATASET", "gsm8k"),
        choices=list(DATASETS.keys()),
        help="Dataset to use",
    )
    parser.add_argument(
        "-c",
        "--constraint",
        default=os.environ.get("CONSTRAINT"),
        choices=list(CONSTRAINTS.keys()),
        help="Reasoning constraint to apply",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name for this eval run (shows in logs). Defaults to '{constraint}_{dataset}'",
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=int(os.environ.get("N_SAMPLES", 10)),
        help="Number of samples from the dataset to evaluate (0 = full dataset)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("MAX_TOKENS", 256)),
        help="Maximum tokens for model generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", 42)),
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("EPOCHS", 1)),
        help="Number of times to run each sample (reduces variance via majority vote)",
    )
    parser.add_argument(
        "--repeat-input",
        type=int,
        default=1,
        help="Number of times to repeat the question in the prompt (single-stage only)",
    )
    parser.add_argument(
        "--filler-tokens",
        type=int,
        default=0,
        help="Number of filler tokens (periods) to add to the assistant message (single-stage only)",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage evaluation: reason first (without answer), then answer",
    )
    parser.add_argument(
        "--force-answer-prefix",
        type=bool,
        default=os.environ.get("FORCE_ANSWER_PREFIX", "").lower() in ("true", "1"),
        help="Force '\\n<answer>' after end tag",
    )
    parser.add_argument(
        "--use-logit-mask",
        type=bool,
        default=os.environ.get("USE_LOGIT_MASK", "").lower() in ("true", "1"),
        help="Use logit masking",
    )
    parser.add_argument(
        "--strip-reasoning",
        action="store_true",
        help="Strip non-emoji characters from reasoning before final answer (requires --two-stage)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for eval logs. Defaults to project 'logs/'.",
    )

    args = parser.parse_args()

    # Validate: single-stage requires constraint
    if not args.two_stage and not args.constraint:
        parser.error(
            "--constraint is required for single-stage mode (or use --two-stage)"
        )

    # Validate: strip-reasoning requires two-stage
    if args.strip_reasoning and not args.two_stage:
        parser.error("--strip-reasoning requires --two-stage")

    # Build eval name
    if args.two_stage:
        constraint_part = args.constraint or "unconstrained"
        if args.strip_reasoning:
            eval_name = (
                args.name
                or f"2stage_stripped_{constraint_part}_{args.dataset}_{short_model_name(args.model)}"
            )
        else:
            eval_name = (
                args.name
                or f"2stage_{constraint_part}_{args.dataset}_{short_model_name(args.model)}"
            )
    else:
        eval_name = (
            args.name
            or f"{args.constraint}_{args.dataset}_{short_model_name(args.model)}"
        )
        if args.repeat_input > 1:
            eval_name = f"{eval_name}_repeat{args.repeat_input}"

    print("Running evaluation...")
    print(f"  Name:       {eval_name}")
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Constraint: {args.constraint or '(none)'}")
    mode_str = "two-stage" if args.two_stage else "single-stage"
    if args.strip_reasoning:
        mode_str += " (strip non-emoji)"
    print(f"  Mode:       {mode_str}")
    print(f"  Samples:    {args.n_samples}")
    print(f"  Max Tokens: {args.max_tokens}")
    print(f"  Epochs:     {args.epochs}")
    if not args.two_stage:
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
        two_stage=args.two_stage,
        strip_reasoning=args.strip_reasoning,
        name=eval_name,
        max_tokens=args.max_tokens,
        force_answer_prefix="\n<answer>" if args.force_answer_prefix else None,
        use_logit_mask=args.use_logit_mask,
        log_dir=args.log_dir,
    )

    return results


if __name__ == "__main__":
    main()
