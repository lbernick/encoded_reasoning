# %%
# Ensure plots display inline in Jupyter
try:
    get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore[name-defined]
except NameError:
    pass

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from inspect_ai.analysis import samples_df

try:
    from IPython.display import display
except ImportError:
    display = None

# %%
SPECIAL_LOGS = Path(__file__).resolve().parent.parent / "special-logs"

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
    ax.bar(x, non_reasoning_acc, width, label="Non-reasoning", color="tab:orange", edgecolor="gray")
    ax.bar(x, gain_acc, width, bottom=non_reasoning_acc, label="Gain from reasoning", color="tab:green", edgecolor="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.set_title("GSM8K: Reasoning vs non-reasoning accuracy by model")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(SPECIAL_LOGS.parent / "gsm8k_reasoning_vs_non_reasoning.png", dpi=150, bbox_inches="tight")
    if display is not None:
        display(fig)
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
    ax.bar(range(len(labels)), recovery, color="tab:blue", edgecolor="gray")
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
    if display is not None:
        display(fig)
    plt.show()


# %%
# Qwen 2.5 14B GSM8K: recovery by intervention
QWEN_GSM8K_14B_FILES: dict[str, str] = {
    "cot": "qwen2.5-14b-gsm8k-cot-subset.eval",
    "no_cot": "qwen2.5-14b-gsm8k-no-cot-subset.eval",
    "repeat1": "qwen2.5-14b-gsm8k-repeat1-subset.eval",
    "repeat2": "qwen2.5-14b-gsm8k-repeat2-subset.eval",
    "repeat3": "qwen2.5-14b-gsm8k-repeat3-subset.eval",
    "repeat4": "qwen2.5-14b-gsm8k-repeat4-subset.eval",
    "repeat5": "qwen2.5-14b-gsm8k-repeat5-subset.eval",
    "repeat10": "qwen2.5-14b-gsm8k-repeat10-subset.eval",
    "filler10": "qwen2.5-14b-gsm8k-filler10-subset.eval",
    "filler100": "qwen2.5-14b-gsm8k-filler100-subset.eval",
    "filler1000": "qwen2.5-14b-gsm8k-filler1000-subset.eval",
}

def _load_acc_from_files(files: dict[str, str], score_col: str = "score_gsm8k_scorer") -> dict[str, float]:
    acc: dict[str, float] = {}
    for name, filename in files.items():
        path = SPECIAL_LOGS / filename
        df = samples_df(logs=str(path), quiet=True)
        if score_col not in df.columns:
            raise ValueError(f"Missing {score_col} in {path}")
        acc[name] = df[score_col].mean()
    return acc

qwen_gsm_acc = _load_acc_from_files(QWEN_GSM8K_14B_FILES)
reasoning_acc = qwen_gsm_acc["cot"]
non_reasoning_acc = qwen_gsm_acc["no_cot"]
interventions = {
    label: qwen_gsm_acc[key]
    for key, label in [
        ("no_cot", "no CoT"),
        ("repeat1", "repeat 1"),
        ("repeat2", "repeat 2"),
        ("repeat3", "repeat 3"),
        ("repeat4", "repeat 4"),
        ("repeat5", "repeat 5"),
        ("repeat10", "repeat 10"),
        ("filler10", "filler 10"),
        ("filler100", "filler 100"),
        ("filler1000", "filler 1000"),
    ]
}
recovery_by_intervention(
    "Qwen 2.5 14B GSM8K",
    reasoning_acc,
    non_reasoning_acc,
    interventions,
    output_path=SPECIAL_LOGS.parent / "qwen_gsm8k_recovery.png",
)

# %%
# GPT 5.2 GSM8K: recovery by repeat
GPT52_GSM8K_FILES: dict[str, str] = {
    "cot": "gpt52/gpt52-gsm8k-cot-subset.eval",
    "repeat1": "gpt52/gpt52-gsm8k-repeat1-subset.eval",
    "repeat2": "gpt52/gpt52-gsm8k-repeat2-subset.eval",
    "repeat3": "gpt52/gpt52-gsm8k-repeat3-subset.eval",
    "repeat5": "gpt52/gpt52-gsm8k-repeat5-subset.eval",
    "repeat10": "gpt52/gpt52-gsm8k-repeat10-subset.eval",
}

gpt52_acc: dict[str, float] = {}
for name, filename in GPT52_GSM8K_FILES.items():
    path = SPECIAL_LOGS / filename
    df = samples_df(logs=str(path), quiet=True)
    if "score_gsm8k_scorer" not in df.columns:
        raise ValueError(f"Missing score_gsm8k_scorer in {path}")
    gpt52_acc[name] = df["score_gsm8k_scorer"].mean()

reasoning_acc = gpt52_acc["cot"]
non_reasoning_acc = gpt52_acc["repeat1"]
interventions = {
    label: gpt52_acc[key]
    for key, label in [
        ("repeat2", "repeat 2"),
        ("repeat3", "repeat 3"),
        ("repeat5", "repeat 5"),
        ("repeat10", "repeat 10"),
    ]
}
recovery_by_intervention(
    "GPT 5.2 GSM8K",
    reasoning_acc,
    non_reasoning_acc,
    interventions,
    output_path=SPECIAL_LOGS.parent / "gpt52_gsm8k_recovery.png",
)
# %%
# GPT 5.2 MoreHopQA: recovery by repeat
GPT52_MOREHOP_FILES: dict[str, str] = {
    "cot": "gpt52/gpt52-morehopqa-cot-subset.eval",
    "no_cot": "gpt52/gpt52-morehopqa-no-cot-subset.eval",
    "repeat2": "gpt52/gpt52-morehopqa-repeat2-subset.eval",
    "repeat3": "gpt52/gpt52-morehopqa-repeat3-subset.eval",
    "repeat5": "gpt52/gpt52-morehopqa-repeat5-subset.eval",
    "repeat10": "gpt52/gpt52-morehopqa-repeat10-subset.eval",
}

MOREHOPQA_SCORE_COL = "score_morehopqa_scorer"

gpt52_morehop_acc: dict[str, float] = {}
for name, filename in GPT52_MOREHOP_FILES.items():
    path = SPECIAL_LOGS / filename
    df = samples_df(logs=str(path), quiet=True)
    if MOREHOPQA_SCORE_COL not in df.columns:
        raise ValueError(f"Missing {MOREHOPQA_SCORE_COL} in {path}")
    gpt52_morehop_acc[name] = df[MOREHOPQA_SCORE_COL].mean()

reasoning_acc = gpt52_morehop_acc["cot"]
non_reasoning_acc = gpt52_morehop_acc["no_cot"]
interventions = {
    label: gpt52_morehop_acc[key]
    for key, label in [
        ("no_cot", "no CoT"),
        ("repeat2", "repeat 2"),
        ("repeat3", "repeat 3"),
        ("repeat5", "repeat 5"),
        ("repeat10", "repeat 10"),
    ]
}
recovery_by_intervention(
    "GPT 5.2 MoreHopQA",
    reasoning_acc,
    non_reasoning_acc,
    interventions,
    output_path=SPECIAL_LOGS.parent / "gpt52_morehopqa_recovery.png",
)
# %% 