#!/usr/bin/env python3
"""
Plot evaluation results with error bars from Inspect logs.

Usage:
    python -m evals.plot_results logs/log1.eval logs/log2.eval logs/log3.eval
    python -m evals.plot_results --pattern "*haiku*gsm8k*"
"""

import argparse
import glob
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log


def extract_results(log_path: str) -> dict:
    """Extract results from an eval log."""
    log = read_eval_log(log_path)

    # Get accuracy from results
    accuracy = None
    for score in log.results.scores:
        if "accuracy" in score.metrics:
            accuracy = score.metrics["accuracy"].value
            break

    # Get scored_samples (post-reduction count for epochs) for correct stderr
    n_scored = None
    stderr = None
    for score in log.results.scores:
        n_scored = score.scored_samples
        # Use stderr from metrics if available
        if "stderr" in score.metrics:
            stderr = score.metrics["stderr"].value
        break

    # Fallback: compute stderr manually using scored_samples (not total samples)
    if stderr is None and n_scored and accuracy is not None:
        stderr = math.sqrt(accuracy * (1 - accuracy) / n_scored)
    elif stderr is None:
        stderr = 0

    n_total = n_scored or len(log.samples)

    # Extract name/metadata
    name = log.eval.task

    # Try to extract repeat value from name
    repeat_match = re.search(r"repeat(\d+)", name)
    repeat = int(repeat_match.group(1)) if repeat_match else 1

    return {
        "name": name,
        "path": log_path,
        "accuracy": accuracy,
        "stderr": stderr,
        "n_samples": n_total,
        "repeat": repeat,
        "model": log.eval.model,
        "dataset": log.eval.dataset.name if log.eval.dataset else "unknown",
    }


def plot_results(results: list[dict], output_path: str | None = None, title: str | None = None):
    """Create bar chart with error bars."""
    # Sort by repeat value
    results = sorted(results, key=lambda x: x["repeat"])

    # Extract data
    labels = [f"repeat={r['repeat']}" for r in results]
    accuracies = [r["accuracy"] for r in results]
    stderrs = [r["stderr"] for r in results]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bar positions
    x = range(len(results))

    # Create bars with error bars
    bars = ax.bar(x, accuracies, yerr=stderrs, capsize=5, color="steelblue", edgecolor="black")

    # Add value labels on bars
    for i, (bar, acc, err) in enumerate(zip(bars, accuracies, stderrs)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err + 0.01,
            f"{acc:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.0)

    # Title
    if title:
        ax.set_title(title, fontsize=14)
    else:
        # Auto-generate title from first result
        r = results[0]
        model_short = r["model"].split("/")[-1]
        ax.set_title(f"{model_short} on {r['dataset']} (n={r['n_samples']})", fontsize=14)

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Add sample size annotation
    n = results[0]["n_samples"]
    ax.annotate(
        f"Error bars: ±1 SE",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results with error bars")
    parser.add_argument("logs", nargs="*", help="Log files to plot")
    parser.add_argument("--pattern", "-p", help="Glob pattern to find logs (e.g., '*haiku*gsm8k*')")
    parser.add_argument("--output", "-o", help="Output file path (e.g., results.png)")
    parser.add_argument("--title", "-t", help="Plot title")

    args = parser.parse_args()

    # Collect log files
    log_files = []

    if args.logs:
        log_files.extend(args.logs)

    if args.pattern:
        pattern = f"logs/{args.pattern}.eval" if not args.pattern.endswith(".eval") else f"logs/{args.pattern}"
        log_files.extend(glob.glob(pattern))

    if not log_files:
        print("No log files specified. Use positional args or --pattern")
        sys.exit(1)

    # Remove duplicates and sort
    log_files = sorted(set(log_files))

    print(f"Found {len(log_files)} log files:")
    for f in log_files:
        print(f"  {f}")
    print()

    # Extract results
    results = []
    for log_file in log_files:
        try:
            result = extract_results(log_file)
            results.append(result)
            print(f"{result['name']}: {result['accuracy']:.1%} ± {result['stderr']:.1%} (n={result['n_samples']})")
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    if not results:
        print("No results extracted")
        sys.exit(1)

    print()

    # Plot
    plot_results(results, output_path=args.output, title=args.title)


if __name__ == "__main__":
    main()
