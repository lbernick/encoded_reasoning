#!/usr/bin/env python3
"""
Filler-token ablation on frontier models (GPT-5.2, Opus 4.5).

Fills in the gap left by qwen2.5-14b_ablations.py, which only ran filler
tokens on Qwen. Uses the single-stage no_cot constraint with N filler
periods prefilled into the assistant message (--filler-tokens), matching
the Qwen setup so recovery numbers are comparable.

Also (re)runs the cot / no_cot baselines in the same batch so the
CoT-ceiling and no-CoT-floor are measured on the identical subset the
filler runs use (avoids the baseline-mismatch issue seen in the GPT-5.2
MoreHopQA numbers).

Run:  python experiments/frontier_filler.py
"""

import os
import subprocess
import sys
from pathlib import Path

SAMPLES = 200
MAX_TOKENS = 512
LOG_DIR = "special-logs"
FILLER_COUNTS = [10, 100, 1000]

MODELS: dict[str, str] = {
    "gpt52": "openrouter/openai/gpt-5.2",
    "opus4.5": "openrouter/anthropic/claude-opus-4.5",
}
DATASETS = ["gsm8k", "morehopqa"]


def run_eval(
    model_id: str,
    dataset: str,
    constraint: str,
    name: str,
    filler_tokens: int = 0,
) -> None:
    cmd = [
        sys.executable, "-m", "encoded_reasoning",
        "--model", model_id,
        "--dataset", dataset,
        "--constraint", constraint,
        "--name", name,
        "-n", str(SAMPLES),
        "--max-tokens", str(MAX_TOKENS),
        "--log-dir", LOG_DIR,
    ]
    if filler_tokens > 0:
        cmd.extend(["--filler-tokens", str(filler_tokens)])

    print(f"\n{'='*60}\nRunning: {name}\nModel: {model_id}\n{'='*60}\n")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {name} failed (exit {e.returncode})")


def main():
    os.chdir(Path(__file__).parent.parent)
    for short, model_id in MODELS.items():
        for dataset in DATASETS:
            # Baselines (matched subset) — ceiling and floor
            run_eval(model_id, dataset, "unconstrained", f"{short}-{dataset}-filler-cot-subset")
            run_eval(model_id, dataset, "no_cot", f"{short}-{dataset}-filler0-subset")
            # Filler-token interventions (single-stage, no_cot + prefilled periods)
            for n in FILLER_COUNTS:
                run_eval(
                    model_id, dataset, "no_cot",
                    f"{short}-{dataset}-filler{n}-subset",
                    filler_tokens=n,
                )
    print("\nDone. Analyze with notebooks/absolute_diffs.py (add the new patterns).")


if __name__ == "__main__":
    main()
