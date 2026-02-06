#!/usr/bin/env python3
"""
Opus 4.5 GSM8K prompt-repeat experiments.

Runs no-CoT with repeated prompts (repeat 2, 3, 5, 10) to measure
how much of the CoT–no-CoT gap is recovered by repeating the question.

Existing data in special-logs/:
  - opus4.5-gsm8k-cot-subset.eval        (CoT baseline)
  - opus4.5-gsm8k-no-cot-subset.eval      (no-CoT = repeat 1)

This script fills in the missing repeat counts.
"""

import subprocess
import sys
from pathlib import Path

MODEL_ID = "openrouter/anthropic/claude-opus-4.5"
MODEL_SHORT = "opus4.5"
SAMPLES = 200
DATASET = "gsm8k"
MAX_TOKENS = 512
LOG_DIR = "special-logs"

# Only the repeats we're missing (1 and CoT already exist)
REPEATS = [2, 3, 5, 10]


def run_eval(repeat_input: int) -> None:
    """Run a single no-CoT evaluation with the given repeat count."""
    name = f"{MODEL_SHORT}-{DATASET}-repeat{repeat_input}-subset"

    cmd = [
        sys.executable, "-m", "encoded_reasoning",
        "--model", MODEL_ID,
        "--dataset", DATASET,
        "--constraint", "no_cot",
        "--name", name,
        "-n", str(SAMPLES),
        "--max-tokens", str(MAX_TOKENS),
        "--repeat-input", str(repeat_input),
    ]
    if LOG_DIR:
        cmd.extend(["--log-dir", LOG_DIR])

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Model: {MODEL_ID}")
    print(f"Repeat: {repeat_input}")
    print(f"Samples: {SAMPLES}")
    print(f"{'='*60}\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {name} failed with exit code {e.returncode}")


def main() -> None:
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)

    print("Opus 4.5 GSM8K Prompt Repeat Experiment")
    print(f"  Model:   {MODEL_ID}")
    print(f"  Dataset: {DATASET}")
    print(f"  Samples: {SAMPLES}")
    print(f"  Repeats: {REPEATS}")
    print(f"  Log dir: {LOG_DIR}")
    print()

    for i, repeat in enumerate(REPEATS, 1):
        print(f"[{i}/{len(REPEATS)}]", end="")
        run_eval(repeat)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
