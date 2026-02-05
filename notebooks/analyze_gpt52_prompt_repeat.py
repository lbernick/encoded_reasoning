# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # GPT-5.2 Prompt Repeat Effects
#
# Analyzes evaluation results for GPT-5.2 under different prompt-repeat and
# reasoning conditions. Compares CoT baseline vs no-CoT with repeated prompts
# (repeat 1, 2, 3, 5, 10) on GSM8K and MoreHopQA.
# Logs are read from `special-logs/gpt52/`.

# %%
import json
import re
import sys
import zipfile
from pathlib import Path

# Use non-interactive backend when run as script so plt.show() doesn't block
if "ipykernel" not in sys.modules:
    import matplotlib
    matplotlib.use("Agg")
    import warnings
    warnings.filterwarnings("ignore", message=".*non-interactive.*cannot be shown")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# %% [markdown]
# ## Configuration

# %%
SPECIAL_LOGS_DIR = Path(__file__).resolve().parent.parent / "special-logs" / "gpt52"

# Expected eval files: gpt52-{dataset}-{condition}-subset.eval
# dataset: gsm8k | morehopqa
# condition: cot | no-cot | emojis | repeat1 | repeat2 | repeat3 | repeat5 | repeat10

# %% [markdown]
# ## Load evaluation results

# %%
def load_eval_file(eval_path: Path) -> list[dict]:
    """Load samples from an eval zip file."""
    samples = []
    with zipfile.ZipFile(eval_path, "r") as z:
        sample_files = [
            f for f in z.namelist()
            if f.startswith("samples/") and f.endswith(".json")
        ]
        for sample_file in sorted(sample_files):
            with z.open(sample_file) as f:
                sample_data = json.load(f)
                samples.append(sample_data)
    return samples


def parse_condition(filename: str) -> tuple[str, str] | None:
    """Parse (dataset, condition) from filename like gpt52-gsm8k-repeat3-subset.eval."""
    match = re.match(r"gpt52-(gsm8k|morehopqa)-(.+)-subset\.eval", filename)
    if not match:
        return None
    dataset, condition = match.groups()
    return dataset, condition


def get_score_value(sample: dict, dataset: str) -> bool | None:
    """Extract correct (bool) from sample scores for the dataset."""
    scores = sample.get("scores", {})
    if dataset == "gsm8k" and "gsm8k_scorer" in scores:
        return scores["gsm8k_scorer"].get("value")
    if dataset == "morehopqa" and "morehopqa_scorer" in scores:
        return scores["morehopqa_scorer"].get("value")
    return None


def extract_data(samples: list[dict], dataset: str) -> list[dict]:
    """Extract key metrics from samples."""
    data = []
    for sample in samples:
        metadata = sample.get("metadata", {})
        correct = get_score_value(sample, dataset)
        if correct is None:
            continue
        output = sample.get("output", {}) or {}
        usage = output.get("usage", {}) or {}
        record = {
            "sample_id": sample.get("id"),
            "question": (sample.get("input") or "")[:100],
            "target": sample.get("target"),
            "correct": correct,
            "model": output.get("model"),
            "tokens": usage.get("total_tokens"),
            "no_of_hops": metadata.get("no_of_hops") if dataset == "morehopqa" else None,
            "reasoning_type": metadata.get("reasoning_type") if dataset == "morehopqa" else None,
        }
        data.append(record)
    return data


def load_all_gpt52_evals() -> dict[str, dict[str, pd.DataFrame]]:
    """Load all gpt52 eval files; return {dataset: {condition: df}}."""
    result: dict[str, dict[str, pd.DataFrame]] = {"gsm8k": {}, "morehopqa": {}}
    if not SPECIAL_LOGS_DIR.exists():
        print(f"Warning: {SPECIAL_LOGS_DIR} not found")
        return result

    for eval_path in sorted(SPECIAL_LOGS_DIR.glob("*.eval")):
        parsed = parse_condition(eval_path.name)
        if not parsed:
            continue
        dataset, condition = parsed
        samples = load_eval_file(eval_path)
        data = extract_data(samples, dataset)
        df = pd.DataFrame(data)
        result[dataset][condition] = df
        print(f"{dataset} / {condition}: {len(df)} samples, accuracy {df['correct'].mean()*100:.1f}%")

    return result


eval_data = load_all_gpt52_evals()

# %% [markdown]
# ## GSM8K: Overall accuracy by condition

# %%
gsm8k_conditions = [
    "cot", "repeat1", "repeat2", "repeat3", "repeat5", "repeat10"
]
gsm8k_order = [c for c in gsm8k_conditions if c in eval_data["gsm8k"]]

print("=== GSM8K: Overall statistics ===")
for cond in gsm8k_order:
    df = eval_data["gsm8k"][cond]
    acc = df["correct"].mean()
    n = len(df)
    correct = df["correct"].sum()
    tokens = df["tokens"].mean()
    print(f"  {cond}: {acc*100:.1f}% ({int(correct)}/{n}), avg tokens {tokens:.0f}")

# %% [markdown]
# ## GSM8K: Bar chart (accuracy vs condition)

# %%
if gsm8k_order:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(gsm8k_order))
    accs = [eval_data["gsm8k"][c]["correct"].mean() for c in gsm8k_order]
    counts = [len(eval_data["gsm8k"][c]) for c in gsm8k_order]
    std_errs = [
        np.sqrt(acc * (1 - acc) / n) if n else 0
        for acc, n in zip(accs, counts)
    ]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(gsm8k_order)))
    bars = ax.bar(x, accs, yerr=std_errs, capsize=4, color=colors, edgecolor="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(gsm8k_order)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Condition")
    ax.set_title("GPT-5.2 on GSM8K: Prompt repeat effects (no CoT)")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=accs[0], color="gray", linestyle="--", alpha=0.7, label="CoT baseline" if gsm8k_order[0] == "cot" else None)
    for i, (acc, n) in enumerate(zip(accs, counts)):
        ax.text(i, acc + std_errs[i] + 0.02, f"n={n}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## MoreHopQA: Overall statistics by condition

# %%
morehop_conditions = [
    "cot", "no-cot", "repeat2", "repeat3", "repeat5", "repeat10", "emojis"
]
morehop_order = [c for c in morehop_conditions if c in eval_data["morehopqa"]]

print("=== MoreHopQA: Overall statistics ===")
for cond in morehop_order:
    df = eval_data["morehopqa"][cond]
    acc = df["correct"].mean()
    n = len(df)
    correct = df["correct"].sum()
    tokens = df["tokens"].mean()
    print(f"  {cond}: {acc*100:.1f}% ({int(correct)}/{n}), avg tokens {tokens:.0f}")

# %% [markdown]
# ## MoreHopQA: Performance by number of hops

# %%
hop_stats_morehop: dict[str, pd.DataFrame] = {}
for cond in morehop_order:
    df = eval_data["morehopqa"][cond]
    df_hops = df[df["no_of_hops"].notna()].copy()
    if df_hops.empty:
        continue
    hop_stats = df_hops.groupby("no_of_hops").agg({
        "correct": ["sum", "count", "mean"],
        "tokens": ["mean", "median", "std"],
    }).round(3)
    hop_stats.columns = ["Correct", "Total", "Accuracy", "Avg Tokens", "Median Tokens", "Std Tokens"]
    hop_stats_morehop[cond] = hop_stats
    print(f"\n=== MoreHopQA / {cond}: by hops ===")
    print(hop_stats)

# %% [markdown]
# ## MoreHopQA: Accuracy by hops (all conditions)

# %%
all_hops_data_morehop: dict[str, pd.DataFrame] = {}
for cond in morehop_order:
    df = eval_data["morehopqa"][cond]
    df_hops = df[df["no_of_hops"].notna()].copy()
    if df_hops.empty:
        continue
    hop_accuracy = df_hops.groupby("no_of_hops")["correct"].agg(["sum", "count"])
    hop_accuracy["accuracy"] = hop_accuracy["sum"] / hop_accuracy["count"]
    hop_accuracy["std_error"] = np.sqrt(
        hop_accuracy["accuracy"] * (1 - hop_accuracy["accuracy"]) / hop_accuracy["count"]
    )
    all_hops_data_morehop[cond] = hop_accuracy

all_hop_counts = sorted(
    set().union(*[set(d.index) for d in all_hops_data_morehop.values()])
)

comparison_data = {}
for hop_count in all_hop_counts:
    row = {}
    for cond in morehop_order:
        if cond in all_hops_data_morehop and hop_count in all_hops_data_morehop[cond].index:
            row[cond] = all_hops_data_morehop[cond].loc[hop_count, "accuracy"]
        else:
            row[cond] = np.nan
    comparison_data[hop_count] = row

comparison_df = pd.DataFrame(comparison_data).T
print("=== MoreHopQA: Accuracy by hops (all conditions) ===")
print(comparison_df.round(4))

# %% [markdown]
# ## MoreHopQA: Bar chart by hops (repeat conditions)

# %%
# Plot conditions that are repeat-related (exclude emojis for clarity if many)
plot_conds = [c for c in morehop_order if c in all_hops_data_morehop]
if plot_conds and all_hop_counts:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_hop_counts))
    width = 0.12
    colors = plt.cm.Set2(np.linspace(0, 1, len(plot_conds)))

    for i, cond in enumerate(plot_conds):
        offset = width * (i - (len(plot_conds) - 1) / 2)
        accs = []
        errs = []
        for hop_count in all_hop_counts:
            if hop_count in all_hops_data_morehop[cond].index:
                acc = all_hops_data_morehop[cond].loc[hop_count, "accuracy"]
                err = all_hops_data_morehop[cond].loc[hop_count, "std_error"]
            else:
                acc, err = np.nan, 0
            accs.append(acc)
            errs.append(err)
        ax.bar(
            x + offset, accs, width, label=cond, color=colors[i], alpha=0.8,
            yerr=errs, capsize=2, error_kw={"linewidth": 1}
        )

    ax.set_xlabel("Number of hops")
    ax.set_ylabel("Accuracy")
    ax.set_title("GPT-5.2 MoreHopQA: Accuracy by hops across conditions")
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in all_hop_counts])
    ax.set_ylim(0, 1.1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## MoreHopQA: Overall accuracy bar chart

# %%
if morehop_order:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(morehop_order))
    accs = [eval_data["morehopqa"][c]["correct"].mean() for c in morehop_order]
    counts = [len(eval_data["morehopqa"][c]) for c in morehop_order]
    std_errs = [
        np.sqrt(acc * (1 - acc) / n) if n else 0
        for acc, n in zip(accs, counts)
    ]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(morehop_order)))
    ax.bar(x, accs, yerr=std_errs, capsize=4, color=colors, edgecolor="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(morehop_order, rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Condition")
    ax.set_title("GPT-5.2 on MoreHopQA: All conditions")
    ax.set_ylim(0, 1.05)
    for i, (acc, n) in enumerate(zip(accs, counts)):
        ax.text(i, acc + std_errs[i] + 0.02, f"n={n}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Summary: Prompt repeat effects on GPT-5.2

# %%
print("=== SUMMARY: GPT-5.2 prompt repeat effects ===\n")

if eval_data["gsm8k"]:
    print("GSM8K:")
    cot_acc = eval_data["gsm8k"].get("cot")
    cot_pct = cot_acc["correct"].mean() * 100 if cot_acc is not None else None
    if cot_pct is not None:
        print(f"  CoT baseline: {cot_pct:.1f}%")
    for cond in gsm8k_order:
        if cond == "cot":
            continue
        df = eval_data["gsm8k"][cond]
        acc = df["correct"].mean() * 100
        diff = (acc - cot_pct) if cot_pct is not None else None
        diff_str = f" ({diff:+.1f} pp vs CoT)" if diff is not None else ""
        print(f"  {cond}: {acc:.1f}%{diff_str}")

if eval_data["morehopqa"]:
    print("\nMoreHopQA:")
    cot_acc = eval_data["morehopqa"].get("cot")
    cot_pct = cot_acc["correct"].mean() * 100 if cot_acc is not None else None
    if cot_pct is not None:
        print(f"  CoT baseline: {cot_pct:.1f}%")
    for cond in morehop_order:
        if cond == "cot":
            continue
        df = eval_data["morehopqa"][cond]
        acc = df["correct"].mean() * 100
        diff = (acc - cot_pct) if cot_pct is not None else None
        diff_str = f" ({diff:+.1f} pp vs CoT)" if diff is not None else ""
        print(f"  {cond}: {acc:.1f}%{diff_str}")
