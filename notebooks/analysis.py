# %%
# Ensure plots display inline in Jupyter
# try:
#     get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore[name-defined]
# except NameError:
#     pass

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from inspect_ai.analysis import samples_df


# %%
SPECIAL_LOGS = Path(__file__).resolve().parent.parent / "special-logs"

# Style guide: GPT blue, Qwen purple, Anthropic (Claude/Opus) orange
def model_color(model_name: str) -> str:
    """Return plot color for a model by vendor/family."""
    m = model_name.lower()
    if "gpt" in m or "openai" in m:
        return "tab:blue"
    if "qwen" in m:
        return "tab:purple"
    if "opus" in m or "claude" in m or "anthropic" in m or "sonnet" in m or "haiku" in m:
        return "tab:orange"
    return "tab:gray"

GSM8K_EVAL_FILES: dict[str, tuple[str, str]] = {
    "gpt52": ("gpt52/gpt52-gsm8k-cot-subset.eval", "gpt52/gpt52-gsm8k-repeat1-subset.eval"),
    "opus4.5": ("opus4.5-gsm8k-cot-subset.eval", "opus4.5-gsm8k-no-cot-subset.eval"),
    "qwen2.5-7b": ("qwen2.5-7b-gsm8k-cot-subset.eval", "qwen2.5-7b-gsm8k-repeat1-subset.eval"),
    "qwen14b": ("qwen14b-gsm8k-cot-subset.eval", "qwen14b-gsm8k-no-cot-subset.eval"),
}

gsm8k_by_model: dict[str, dict[str, pd.DataFrame]] = {}
for model, (reasoning_file, non_reasoning_file) in GSM8K_EVAL_FILES.items():
    reasoning_path = SPECIAL_LOGS / reasoning_file
    non_reasoning_path = SPECIAL_LOGS / non_reasoning_file
    df_r = samples_df(logs=str(reasoning_path), quiet=True)
    if "score_gsm8k_scorer" not in df_r.columns:
        raise ValueError(f"Missing score_gsm8k_scorer in {reasoning_path}")
    out_reasoning = df_r[["sample_id", "input", "target", "score_gsm8k_scorer", "total_tokens"]].rename(
        columns={"score_gsm8k_scorer": "correct", "total_tokens": "tokens"}
    )
    df_n = samples_df(logs=str(non_reasoning_path), quiet=True)
    if "score_gsm8k_scorer" not in df_n.columns:
        raise ValueError(f"Missing score_gsm8k_scorer in {non_reasoning_path}")
    out_non_reasoning = df_n[["sample_id", "input", "target", "score_gsm8k_scorer", "total_tokens"]].rename(
        columns={"score_gsm8k_scorer": "correct", "total_tokens": "tokens"}
    )
    gsm8k_by_model[model] = {"reasoning": out_reasoning, "non_reasoning": out_non_reasoning}

# %%
# Stacked bar: reasoning vs non-reasoning accuracy by model (export for report)
models = []
non_reasoning_acc = []
gain_acc = []
for model, dfs in gsm8k_by_model.items():
    r_df, n_df = dfs["reasoning"], dfs["non_reasoning"]
    if r_df.empty or n_df.empty:
        raise ValueError(f"Empty reasoning or non_reasoning data for model {model}")
    r_acc = r_df["correct"].mean()
    n_acc = n_df["correct"].mean()
    models.append(model)
    non_reasoning_acc.append(n_acc)
    gain_acc.append(max(0, r_acc - n_acc))

if models:
    x = range(len(models))
    width = 0.6
    fig, ax = plt.subplots(figsize=(10, 5))
    from matplotlib.patches import Patch

    for i, model in enumerate(models):
        c = model_color(model)
        ax.bar(i, non_reasoning_acc[i], width, color=c, edgecolor="gray", alpha=0.85)
        ax.bar(i, gain_acc[i], width, bottom=non_reasoning_acc[i], color=c, edgecolor="gray")
    ax.legend(handles=[Patch(facecolor=model_color(m), edgecolor="gray", label=m) for m in models], loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.set_title("GSM8K: Reasoning vs non-reasoning accuracy by model")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(SPECIAL_LOGS.parent / "gsm8k_reasoning_vs_non_reasoning.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
def prop_recovered(
    intervention_prop: float, reasoning_prop: float, no_reasoning_prop: float
) -> float:
    """Proportion of the reasoning–no-reasoning gap recovered by the intervention."""
    gap = reasoning_prop - no_reasoning_prop
    if gap <= 0:
        return 0.0
    return (intervention_prop - no_reasoning_prop) / gap


def recovery_by_intervention(
    model_name: str,
    reasoning_acc: float,
    non_reasoning_acc: float,
    interventions: dict[str, float],
    *,
    title: str | None = None,
    output_path: Path | str | None = None,
) -> None:
    """
    Plot recovery (prop. of gap recovered) for each intervention.

    recovery = (intervention_acc - non_reasoning_acc) / (reasoning_acc - non_reasoning_acc).
    Bar labels and order follow the keys of interventions (dict order preserved).
    """
    labels = list(interventions.keys())
    recovery = [
        prop_recovered(acc, reasoning_acc, non_reasoning_acc)
        for acc in interventions.values()
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(labels)), recovery, color=model_color(model_name), edgecolor="gray")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Recovery (prop. of gap recovered)")
    ax.set_xlabel("Intervention")
    ax.set_title(
        title or f"{model_name}: Recovery from no-reasoning by intervention"
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# # First prompt repeat most important

# %%
# Recovery by number of prompt repeats: Opus 4.5 vs GPT-5.2 on GSM8K
#
# For each model we need:
#   - CoT accuracy (the ceiling)
#   - No-CoT / repeat-1 accuracy (the floor)
#   - Accuracy at each repeat count (the interventions)
#
# GPT-5.2 files live in special-logs/gpt52/
# Opus 4.5 files live in special-logs/ (may have timestamp prefixes)


def find_eval_file(directory: Path, name_pattern: str) -> Path | None:
    """Find the latest eval file whose name contains *name_pattern*.

    Handles both clean names (``foo.eval``) and timestamp-prefixed names
    (``2026-…_foo_hash.eval``).  Returns the most-recently-modified match.
    """
    candidates = [f for f in directory.glob("*.eval") if name_pattern in f.name]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_gsm8k_accuracy(eval_path: Path) -> float:
    """Return mean accuracy from a GSM8K eval file."""
    df = samples_df(logs=str(eval_path), quiet=True)
    if "score_gsm8k_scorer" not in df.columns:
        raise ValueError(f"Missing score_gsm8k_scorer in {eval_path}")
    return float(df["score_gsm8k_scorer"].mean())


def load_morehopqa_accuracy(eval_path: Path) -> float:
    """Return mean accuracy from a MoreHopQA eval file."""
    df = samples_df(logs=str(eval_path), quiet=True)
    if "score_morehopqa_scorer" not in df.columns:
        raise ValueError(f"Missing score_morehopqa_scorer in {eval_path}")
    return float(df["score_morehopqa_scorer"].mean())


def load_morehopqa_df(eval_path: Path) -> pd.DataFrame:
    """Load MoreHopQA eval with correct and no_of_hops (from metadata)."""
    df = samples_df(logs=str(eval_path), quiet=True)
    if "score_morehopqa_scorer" not in df.columns:
        raise ValueError(f"Missing score_morehopqa_scorer in {eval_path}")
    out = df[["sample_id", "score_morehopqa_scorer"]].rename(
        columns={"score_morehopqa_scorer": "correct"}
    )
    if "metadata" in df.columns:
        out["no_of_hops"] = df["metadata"].apply(
            lambda m: m.get("no_of_hops") if isinstance(m, dict) else None
        )
    else:
        out["no_of_hops"] = None
    return out


def load_gpt52_morehop_from_zip(eval_path: Path) -> pd.DataFrame:
    """Load GPT-5.2 MoreHopQA eval from zip (same as analyze_gpt52_prompt_repeat)."""
    samples: list[dict] = []
    with zipfile.ZipFile(eval_path, "r") as z:
        for name in sorted(f for f in z.namelist() if f.startswith("samples/") and f.endswith(".json")):
            with z.open(name) as f:
                samples.append(json.load(f))
    rows: list[dict] = []
    for s in samples:
        scores = s.get("scores", {})
        correct = scores.get("morehopqa_scorer", {}).get("value")
        if correct is None:
            continue
        metadata = s.get("metadata", {})
        rows.append({"correct": correct, "no_of_hops": metadata.get("no_of_hops")})
    return pd.DataFrame(rows)


# --- GPT-5.2 ---
GPT52_DIR = SPECIAL_LOGS / "gpt52"
GPT52_COT_PATTERN = "gpt52-gsm8k-cot-subset"
GPT52_REPEATS: dict[int, str] = {
    1: "gpt52-gsm8k-repeat1-subset",
    2: "gpt52-gsm8k-repeat2-subset",
    3: "gpt52-gsm8k-repeat3-subset",
    5: "gpt52-gsm8k-repeat5-subset",
    10: "gpt52-gsm8k-repeat10-subset",
}

# --- Opus 4.5 ---
OPUS_DIR = SPECIAL_LOGS
OPUS_COT_PATTERN = "opus4.5-gsm8k-cot-subset"
OPUS_REPEATS: dict[int, str] = {
    1: "opus4.5-gsm8k-no-cot-subset",
    2: "opus4.5-gsm8k-repeat2-subset",
    3: "opus4.5-gsm8k-repeat3-subset",
    5: "opus4.5-gsm8k-repeat5-subset",
    10: "opus4.5-gsm8k-repeat10-subset",
}

# --- Qwen 2.5 7B ---
QWEN27B_DIR = SPECIAL_LOGS
QWEN27B_COT_PATTERN = "qwen2.5-7b-gsm8k-cot-subset"
QWEN27B_REPEATS: dict[int, str] = {
    1: "qwen2.5-7b-gsm8k-repeat1-subset",
    2: "qwen2.5-7b-gsm8k-repeat2-subset",
    3: "qwen2.5-7b-gsm8k-repeat3-subset",
    5: "qwen2.5-7b-gsm8k-repeat5-subset",
    10: "qwen2.5-7b-gsm8k-repeat10-subset",
}

# Load data for each model
repeat_recovery: dict[str, dict[int, float]] = {}  # model -> {n_repeats: recovery}

for model_label, log_dir, cot_pattern, repeat_patterns in [
    ("GPT-5.2", GPT52_DIR, GPT52_COT_PATTERN, GPT52_REPEATS),
    ("Opus 4.5", OPUS_DIR, OPUS_COT_PATTERN, OPUS_REPEATS),
    ("Qwen 2.5 7B", QWEN27B_DIR, QWEN27B_COT_PATTERN, QWEN27B_REPEATS),
]:
    cot_path = find_eval_file(log_dir, cot_pattern)
    if cot_path is None:
        print(f"WARNING: CoT baseline not found for {model_label} (pattern={cot_pattern})")
        continue
    cot_acc = load_gsm8k_accuracy(cot_path)

    # Repeat-1 is the no-reasoning floor
    r1_path = find_eval_file(log_dir, repeat_patterns[1])
    if r1_path is None:
        print(f"WARNING: Repeat-1 file not found for {model_label}")
        continue
    no_cot_acc = load_gsm8k_accuracy(r1_path)

    print(f"{model_label}: CoT={cot_acc:.3f}, no-CoT={no_cot_acc:.3f}")

    recoveries: dict[int, float] = {}
    for n_repeat, pattern in repeat_patterns.items():
        path = find_eval_file(log_dir, pattern)
        if path is None:
            print(f"  repeat {n_repeat}: file not found ({pattern})")
            continue
        acc = load_gsm8k_accuracy(path)
        rec = prop_recovered(acc, cot_acc, no_cot_acc)
        recoveries[n_repeat] = rec
        print(f"  repeat {n_repeat}: acc={acc:.3f}, recovery={rec:.3f}")

    if recoveries:
        repeat_recovery[model_label] = recoveries

# %%
# Plot: recovery vs number of repeats for each model
if repeat_recovery:
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"GPT-5.2": "o", "Opus 4.5": "s", "Qwen 2.5 7B": "D"}

    for model_label, recoveries in repeat_recovery.items():
        xs = sorted(recoveries.keys())
        ys = [recoveries[x] for x in xs]
        ax.plot(
            xs, ys,
            marker=markers.get(model_label, "^"),
            color=model_color(model_label),
            label=model_label,
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Number of prompt copies")
    ax.set_ylabel("Recovery (prop. of CoT–no-CoT gap)")
    ax.set_title("Recovery by Prompt Copies (GSM8K, n=200)")
    ax.set_ylim(-0.05, 1.1)
    # 0 and 1.0: dotted/dashed only, no solid major grid behind them
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Full recovery")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)

    from matplotlib.ticker import FixedLocator, NullLocator

    # X: major ticks only at prompt-copy values we actually tested (vertical major grid)
    xticks = [x for x in sorted({x for r in repeat_recovery.values() for x in r})]
    ax.set_xticks(xticks)
    ax.xaxis.set_minor_locator(NullLocator())

    # Y: major ticks (and labels) at 0, 0.2, 0.4, 0.6, 0.8, 1; grid only at 0.2–0.8
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # 5 minor lines between each major (step 0.2/6 = 1/30), excluding 0 and 1
    minor_ys = [i / 30 for i in range(1, 30) if i % 6 != 0]
    ax.yaxis.set_minor_locator(FixedLocator(minor_ys))

    # Vertical major at prompt-copy values; horizontal major at 0.2–0.8, minor at 1/30 steps
    ax.grid(axis="x", which="major", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", which="major", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.6, alpha=0.3)
    ax.grid(axis="x", which="minor", visible=False)
    # Hide solid grid at 0 and 1 so only the dotted/dashed axhlines show there
    y_major_gridlines = list(ax.yaxis.get_gridlines())
    if len(y_major_gridlines) >= 2:
        y_major_gridlines[0].set_visible(False)
        y_major_gridlines[-1].set_visible(False)

    ax.tick_params(axis="both", which="minor", length=0)
    ax.legend(loc="upper right", framealpha=1.0)
    plt.tight_layout()
    plt.savefig(SPECIAL_LOGS.parent / "gsm8k_repeat_recovery.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No repeat recovery data available — run experiments first.")

# %% [markdown]
# ## MoreHopQA: Recovery by prompt copies
#
# Uses the same dataset (``morehopqa``, same seed) for all models. Opus
# ``opus-morehop-correct-*`` evals use a different subset and are ignored;
# use ``opus4.5-morehopqa-*-subset`` evals (from ``experiments/opus_morehop_repeats.py``).

# %%
# MoreHopQA repeat data: GPT-5.2 (gpt52/) and Opus 4.5 (same subset, opus4.5-morehopqa-*-subset)
GPT52_MOREHOP_COT = "gpt52-morehopqa-cot-subset"
GPT52_MOREHOP_REPEATS: dict[int, str] = {
    1: "gpt52-morehopqa-no-cot-subset",
    2: "gpt52-morehopqa-repeat2-subset",
    3: "gpt52-morehopqa-repeat3-subset",
    5: "gpt52-morehopqa-repeat5-subset",
    10: "gpt52-morehopqa-repeat10-subset",
}
OPUS_MOREHOP_COT = "opus4.5-morehopqa-cot-subset"
OPUS_MOREHOP_REPEATS: dict[int, str] = {
    1: "opus4.5-morehopqa-no-cot-subset",
    2: "opus4.5-morehopqa-repeat2-subset",
    3: "opus4.5-morehopqa-repeat3-subset",
    5: "opus4.5-morehopqa-repeat5-subset",
    10: "opus4.5-morehopqa-repeat10-subset",
}

repeat_recovery_morehop: dict[str, dict[int, float]] = {}

for model_label, log_dir, cot_pattern, repeat_patterns in [
    ("GPT-5.2", GPT52_DIR, GPT52_MOREHOP_COT, GPT52_MOREHOP_REPEATS),
    ("Opus 4.5", OPUS_DIR, OPUS_MOREHOP_COT, OPUS_MOREHOP_REPEATS),
]:
    cot_path = find_eval_file(log_dir, cot_pattern)
    if cot_path is None:
        print(f"WARNING: MoreHopQA CoT not found for {model_label} (pattern={cot_pattern})")
        continue
    cot_acc = load_morehopqa_accuracy(cot_path)
    r1_path = find_eval_file(log_dir, repeat_patterns[1])
    if r1_path is None:
        print(f"WARNING: MoreHopQA repeat-1 not found for {model_label}")
        continue
    no_cot_acc = load_morehopqa_accuracy(r1_path)
    print(f"{model_label} MoreHopQA: CoT={cot_acc:.3f}, no-CoT={no_cot_acc:.3f}")
    recoveries_m: dict[int, float] = {}
    for n_repeat, pattern in repeat_patterns.items():
        path = find_eval_file(log_dir, pattern)
        if path is None:
            print(f"  repeat {n_repeat}: file not found ({pattern})")
            continue
        acc = load_morehopqa_accuracy(path)
        rec = prop_recovered(acc, cot_acc, no_cot_acc)
        recoveries_m[n_repeat] = rec
        print(f"  repeat {n_repeat}: acc={acc:.3f}, recovery={rec:.3f}")
    if recoveries_m:
        repeat_recovery_morehop[model_label] = recoveries_m


if repeat_recovery_morehop:
    from matplotlib.ticker import FixedLocator, NullLocator

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"GPT-5.2": "o", "Opus 4.5": "s", "Qwen 2.5 7B": "D"}
    for model_label, recoveries in repeat_recovery_morehop.items():
        xs = sorted(recoveries.keys())
        ys = [recoveries[x] for x in xs]
        ax.plot(
            xs, ys,
            marker=markers.get(model_label, "^"),
            color=model_color(model_label),
            label=model_label,
            linewidth=2,
            markersize=8,
        )
    ax.set_xlabel("Number of prompt copies")
    ax.set_ylabel("Recovery (prop. of CoT–no-CoT gap)")
    ax.set_title("MoreHopQA: Recovery by prompt copies")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Full recovery")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    xticks_m = [x for x in sorted({x for r in repeat_recovery_morehop.values() for x in r})]
    ax.set_xticks(xticks_m)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    minor_ys = [i / 30 for i in range(1, 30) if i % 6 != 0]
    ax.yaxis.set_minor_locator(FixedLocator(minor_ys))
    ax.grid(axis="x", which="major", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", which="major", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.6, alpha=0.3)
    ax.grid(axis="x", which="minor", visible=False)
    y_major_gridlines = list(ax.yaxis.get_gridlines())
    if len(y_major_gridlines) >= 2:
        y_major_gridlines[0].set_visible(False)
        y_major_gridlines[-1].set_visible(False)
    ax.tick_params(axis="both", which="minor", length=0)
    ax.legend(loc="upper right", framealpha=1.0)
    plt.tight_layout()
    plt.savefig(SPECIAL_LOGS.parent / "morehopqa_repeat_recovery.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No MoreHopQA repeat recovery data available.")

# %% [markdown]
# ## MoreHopQA: Recovery by hops across repeats (GPT-5.2)
#
# Same layout as the accuracy-by-hops chart in `analyze_gpt52_prompt_repeat.py`,
# but Y-axis is recovery (prop. of CoT–no-CoT gap). CoT and no-CoT baselines
# vary by hop count, so recovery is computed per hop.

# %%
# GPT-5.2 MoreHopQA: load per-condition evals with hop info, compute recovery per hop
MOREHOP_CONDITION_ORDER = ["repeat1", "repeat2", "repeat3", "repeat5", "repeat10"]
morehop_recovery_by_hops: dict[int, dict[str, float]] = {}  # hop -> condition -> recovery
morehop_recovery_se_by_hops: dict[int, dict[str, float]] = {}  # hop -> condition -> std err (optional)

cot_path_gpt52 = find_eval_file(GPT52_DIR, GPT52_MOREHOP_COT)
r1_path_gpt52 = find_eval_file(GPT52_DIR, GPT52_MOREHOP_REPEATS[1]) if cot_path_gpt52 else None

if cot_path_gpt52 and r1_path_gpt52:
    cot_df = load_gpt52_morehop_from_zip(cot_path_gpt52)
    no_cot_df = load_gpt52_morehop_from_zip(r1_path_gpt52)
    if not cot_df["no_of_hops"].isna().all():
        hop_counts = sorted(cot_df["no_of_hops"].dropna().unique())
        condition_dfs: dict[str, pd.DataFrame] = {
            "repeat1": no_cot_df,
        }
        for n_rep in [2, 3, 5, 10]:
            path = find_eval_file(GPT52_DIR, GPT52_MOREHOP_REPEATS[n_rep])
            if path is not None:
                condition_dfs[f"repeat{n_rep}"] = load_gpt52_morehop_from_zip(path)

        for h in hop_counts:
            cot_h = cot_df[cot_df["no_of_hops"] == h]["correct"]
            no_cot_h = no_cot_df[no_cot_df["no_of_hops"] == h]["correct"]
            n_h = len(cot_h)
            if n_h == 0:
                continue
            cot_acc_h = float(cot_h.mean())
            no_cot_acc_h = float(no_cot_h.mean())
            gap_h = cot_acc_h - no_cot_acc_h
            if gap_h <= 0:
                continue
            morehop_recovery_by_hops[int(h)] = {}
            morehop_recovery_se_by_hops[int(h)] = {}
            for cond_name, c_df in condition_dfs.items():
                c_h = c_df[c_df["no_of_hops"] == h]["correct"]
                if len(c_h) == 0:
                    continue
                acc_h = float(c_h.mean())
                rec = prop_recovered(acc_h, cot_acc_h, no_cot_acc_h)
                morehop_recovery_by_hops[int(h)][cond_name] = rec
                # SE of recovery ≈ SE(acc) / gap; SE(acc) = sqrt(acc*(1-acc)/n)
                se_acc = np.sqrt(acc_h * (1 - acc_h) / len(c_h))
                morehop_recovery_se_by_hops[int(h)][cond_name] = se_acc / gap_h
else:
    hop_counts = []

# %%
if morehop_recovery_by_hops:
    all_hop_counts = sorted(morehop_recovery_by_hops.keys())
    plot_conds = [c for c in MOREHOP_CONDITION_ORDER if any(c in morehop_recovery_by_hops[h] for h in all_hop_counts)]
    if plot_conds and all_hop_counts:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(all_hop_counts))
        width = 0.12
        colors = plt.cm.Set2(np.linspace(0, 1, len(plot_conds)))

        for i, cond in enumerate(plot_conds):
            offset = width * (i - (len(plot_conds) - 1) / 2)
            recs = []
            errs = []
            for hop_count in all_hop_counts:
                if cond in morehop_recovery_by_hops.get(hop_count, {}):
                    recs.append(morehop_recovery_by_hops[hop_count][cond])
                    errs.append(morehop_recovery_se_by_hops.get(hop_count, {}).get(cond, 0))
                else:
                    recs.append(np.nan)
                    errs.append(0)
            ax.bar(
                x + offset, recs, width, label=cond, color=colors[i], alpha=0.8,
                yerr=errs, capsize=2, error_kw={"linewidth": 1}
            )

        ax.set_xlabel("Number of hops")
        ax.set_ylabel("Recovery (prop. of CoT–no-CoT gap)")
        ax.set_title("GPT-5.2 MoreHopQA: Recovery by hops across repeats")
        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in all_hop_counts])
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(SPECIAL_LOGS.parent / "morehopqa_recovery_by_hops_gpt52.png", dpi=150, bbox_inches="tight")
        plt.show()
else:
    print("No GPT-5.2 MoreHopQA per-hop data (need evals with metadata.no_of_hops).")

# %%
