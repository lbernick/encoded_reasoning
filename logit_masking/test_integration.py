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

from evals.runner import run_eval
from evals.constraints import CONSTRAINTS
from evals.datasets import DATASETS


def main():
    parser = argparse.ArgumentParser(
        description="Integration test: logit-masked eval through full inspect pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="openai-community/gpt2",
        help="HuggingFace model name (without hf/ prefix — added automatically)",
    )
    parser.add_argument(
        "-d", "--dataset",
        default="gsm8k",
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "-c", "--constraint",
        default="only_emojis",
        choices=list(CONSTRAINTS.keys()),
    )
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    model = f"hf/{args.model}"
    name = f"test_integration_{args.constraint}_{args.dataset}"

    print("Integration test — logit-masked eval")
    print(f"  Model:      {model}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Constraint: {args.constraint}")
    print(f"  Samples:    {args.n_samples}")
    print()

    results = run_eval(
        constraint_name=args.constraint,
        model=model,
        dataset_name=args.dataset,
        n_samples=args.n_samples,
        name=name,
    )

    # Print basic results
    for log in results:
        print(f"Task: {log.eval.task}")
        print(f"Model: {log.eval.model}")
        if log.results:
            for metric in log.results.scores[0].metrics.values():
                print(f"  {metric.name}: {metric.value}")
        if log.samples:
            for sample in log.samples:
                print(f"\n--- Sample ---")
                print(f"Input:  {sample.input[:120]}...")
                print(f"Target: {sample.target}")
                print(f"Output: {sample.output.completion[:300]}")


if __name__ == "__main__":
    main()
