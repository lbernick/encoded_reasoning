"""
CLI entry point for plotting evaluation results.

Usage:
    python -m evals.plotting haiku
    python -m evals.plotting sonnet
    python -m evals.plotting sonnet --output sonnet_results.png
"""

import argparse
from .all_experiments import plot_all_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Plot evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model",
        choices=["haiku", "sonnet"],
        help="Model to plot results for",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["gsm8k", "gpqa"],
        default="gsm8k",
        help="Dataset to plot results for",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (defaults to {model}_{dataset}_all_experiments.png)",
    )

    args = parser.parse_args()

    plot_all_experiments(args.model, args.dataset, args.output)


if __name__ == "__main__":
    main()
