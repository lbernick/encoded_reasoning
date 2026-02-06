#!/usr/bin/env python3
"""
Opus 4.5 MoreHopQA prompt-repeat experiments (same subset as GPT-5.2).

Runs on dataset ``morehopqa`` (not morehopqa_opus_correct) with the same
seed so the question subset is comparable. Produces evals named
opus4.5-morehopqa-*-subset so they can be plotted alongside GPT-5.2
in the MoreHopQA recovery-by-repeats chart.

Run this to get Opus numbers on the MoreHopQA recovery graph.
"""

import subprocess
import sys
from pathlib import Path

MODEL_ID = "openrouter/anthropic/claude-opus-4.5"
MODEL_SHORT = "opus4.5"
SAMPLES = 200
DATASET = "morehopqa"
MAX_TOKENS = 512
LOG_DIR = "special-logs"
SEED = 42

# CoT baseline, then no-CoT at repeat 1, 2, 3, 5, 10
REPEATS = [1, 2, 3, 5, 10]


def run_cot_baseline() -> None:
    """Run CoT (unconstrained) baseline."""
    name = f"{MODEL_SHORT}-{DATASET}-cot-subset"
    cmd = [
        sys.executable, "-m", "encoded_reasoning",
        "--model", MODEL_ID,
        "--dataset", DATASET,
        "--constraint", "unconstrained",
        "--name", name,
        "-n", str(SAMPLES),
        "--max-tokens", str(MAX_TOKENS),
        "--seed", str(SEED),
    ]
    if LOG_DIR:
        cmd.extend(["--log-dir", LOG_DIR])
    print(f"\n{'='*60}\nRunning CoT baseline: {name}\n{'='*60}\n")
    subprocess.run(cmd, check=True)


def run_no_cot(repeat_input: int) -> None:
    """Run no-CoT with the given repeat count."""
    if repeat_input == 1:
        name = f"{MODEL_SHORT}-{DATASET}-no-cot-subset"
    else:
        name = f"{MODEL_SHORT}-{DATASET}-repeat{repeat_input}-subset"
    cmd = [
        sys.executable, "-m", "encoded_reasoning",
        "--model", MODEL_ID,
        "--dataset", DATASET,
        "--constraint", "no_cot",
        "--name", name,
        "-n", str(SAMPLES),
        "--max-tokens", str(MAX_TOKENS),
        "--seed", str(SEED),
        "--repeat-input", str(repeat_input),
    ]
    if LOG_DIR:
        cmd.extend(["--log-dir", LOG_DIR])
    print(f"\n{'='*60}\nRunning: {name} (repeat={repeat_input})\n{'='*60}\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)

    print("Opus 4.5 MoreHopQA Prompt Repeat Experiment (same subset as GPT-5.2)")
    print(f"  Model:   {MODEL_ID}")
    print(f"  Dataset: {DATASET} (seed={SEED})")
    print(f"  Samples: {SAMPLES}")
    print(f"  Log dir: {LOG_DIR}")
    print()

    run_cot_baseline()
    for i, repeat in enumerate(REPEATS, 1):
        print(f"[{i}/{len(REPEATS)}]", end="")
        run_no_cot(repeat)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
