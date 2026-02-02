"""
Create a single comprehensive chart with all experiments for easy comparison.
"""

import glob
import math
import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log
from matplotlib.patches import Patch


def categorize_experiment(filename: str) -> tuple[str, int]:
    """
    Return (label, sort_order) for the experiment.
    Sort order groups experiments logically.
    """
    if "unconstrained" in filename:
        return ("Unconstrained", 1)
    elif "emojis" in filename:
        return ("Only Emojis", 2)
    elif "repeat5" in filename or "no-cot-repeat5" in filename:
        return ("Repeat 5x", 5)
    elif "repeat3" in filename or "no-cot-repeat3" in filename:
        return ("Repeat 3x", 4)
    elif "repeat1" in filename or ("no-cot" in filename and "repeat" not in filename):
        return ("No COT", 3)  # repeat=1 or just no-cot is vanilla no cot
    elif "fill0" in filename:
        return ("Filler 0", 6)
    elif "fill10" in filename:
        return ("Filler 10", 7)
    elif "fill100" in filename:
        return ("Filler 100", 8)
    elif "fill500" in filename:
        return ("Filler 500", 9)
    else:
        return ("Unknown", 99)


def plot_all_experiments(model: str, dataset: str = "gsm8k", output_path: str | None = None):
    """
    Create a comprehensive comparison chart for all experiments.

    Args:
        model: Model name (e.g., "haiku", "sonnet")
        dataset: Dataset name (e.g., "gsm8k", "gpqa")
        output_path: Optional output path. Defaults to {model}_{dataset}_all_experiments.png
    """
    # Map dataset names to log patterns
    dataset_short = "gsm" if dataset == "gsm8k" else dataset

    # Find all relevant log files
    patterns = [
        f"logs/*{model}-{dataset_short}-repeat*.eval",
        f"logs/*{model}-{dataset_short}-no-cot*.eval",
        f"logs/*{model}-{dataset_short}-fill*.eval",
        f"logs/*{model}-{dataset_short}-emojis*.eval",
        f"logs/*{model}-{dataset_short}-unconstrained*.eval",
    ]

    log_files = []
    for pattern in patterns:
        log_files.extend(glob.glob(pattern))

    if not log_files:
        print(f"No log files found for model '{model}'")
        return

    # Group log files by experiment type and keep only the most recent
    from collections import defaultdict
    log_groups = defaultdict(list)
    for log_file in log_files:
        label, sort_order = categorize_experiment(log_file)
        log_groups[label].append(log_file)

    # Keep only the most recent log for each experiment type (sorted by filename timestamp)
    unique_logs = []
    for label, logs in log_groups.items():
        most_recent = sorted(logs)[-1]  # Filenames have timestamps, so last is most recent
        unique_logs.append(most_recent)

    results = []
    for log_file in unique_logs:
        try:
            log = read_eval_log(log_file)

            # Skip if no results yet
            if log.results is None or not log.results.scores:
                print(f"Skipping {log_file} - no results yet")
                continue

            # Get accuracy
            accuracy = None
            stderr = None
            n_scored = None
            for score in log.results.scores:
                if "accuracy" in score.metrics:
                    accuracy = score.metrics["accuracy"].value
                if "stderr" in score.metrics:
                    stderr = score.metrics["stderr"].value
                n_scored = score.scored_samples
                break

            # Compute stderr if not available
            if stderr is None and n_scored and accuracy is not None:
                stderr = math.sqrt(accuracy * (1 - accuracy) / n_scored)
            elif stderr is None:
                stderr = 0
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue

        label, sort_order = categorize_experiment(log_file)

        results.append({
            "label": label,
            "sort_order": sort_order,
            "accuracy": accuracy,
            "stderr": stderr,
            "n_samples": n_scored or len(log.samples),
            "path": log_file,
        })

        print(f"{label:20s}: {accuracy:.1%} ± {stderr:.1%} (n={results[-1]['n_samples']})")

    if not results:
        print("No completed experiments found")
        return

    # Sort by sort_order
    results = sorted(results, key=lambda x: x["sort_order"])

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(results))
    accuracies = [r["accuracy"] for r in results]
    stderrs = [r["stderr"] for r in results]
    labels = [r["label"] for r in results]

    # Color bars by category
    colors = []
    for r in results:
        if r["sort_order"] <= 2:  # Unconstrained, Emojis
            colors.append("steelblue")
        elif r["sort_order"] <= 5:  # No COT and repeats
            colors.append("coral")
        else:  # Fillers
            colors.append("lightseagreen")

    bars = ax.bar(x, accuracies, yerr=stderrs, capsize=5, color=colors, edgecolor="black", alpha=0.8)

    # Add value labels on bars
    for i, (bar, acc, err) in enumerate(zip(bars, accuracies, stderrs)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err + 0.01,
            f"{acc:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=45, ha="right")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{model.capitalize()} 4.5 on {dataset.upper()}: All Experiments Comparison", fontsize=14)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Add legend for colors
    legend_elements = [
        Patch(facecolor="steelblue", edgecolor="black", label="Reasoning Constraints"),
        Patch(facecolor="coral", edgecolor="black", label="Input Repetition"),
        Patch(facecolor="lightseagreen", edgecolor="black", label="Filler Tokens"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    ax.annotate(
        f"Error bars: ±1 SE (n={results[0]['n_samples']})",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()

    if output_path is None:
        output_path = f"{model}_{dataset}_all_experiments.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")


def main():
    import sys

    # Allow model and dataset to be specified as command line arguments
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = "haiku"

    if len(sys.argv) > 2:
        dataset = sys.argv[2]
    else:
        dataset = "gsm8k"

    plot_all_experiments(model, dataset)


if __name__ == "__main__":
    main()
