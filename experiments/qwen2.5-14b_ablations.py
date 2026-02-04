#!/usr/bin/env python3
"""
Qwen 2.5 14B ablation experiments.

Tests:
- No CoT vs CoT (unconstrained)
- Prompt repeats: 2, 3, 4, 5, 10
- Filler tokens: 10, 100, 1000

Uses local HuggingFace model with n=100 samples.
"""

import subprocess
import sys
from pathlib import Path

# Configuration
MODEL = "hf/qwen/qwen2.5-14b-instruct"
MODEL_SHORT = "qwen14b"
SAMPLES = 100
DATASET = "gsm8k"
MAX_TOKENS = 512
LOG_DIR = "special-logs"


def run_eval(
    constraint: str,
    name: str,
    repeat_input: int = 1,
    filler_tokens: int = 0,
) -> None:
    """Run a single evaluation."""
    cmd = [
        sys.executable, "-m", "encoded_reasoning",
        "--model", MODEL,
        "--dataset", DATASET,
        "--constraint", constraint,
        "--name", name,
        "-n", str(SAMPLES),
        "--max-tokens", str(MAX_TOKENS),
        "--log-dir", LOG_DIR,
    ]

    if repeat_input > 1:
        cmd.extend(["--repeat-input", str(repeat_input)])

    if filler_tokens > 0:
        cmd.extend(["--filler-tokens", str(filler_tokens)])

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    subprocess.run(cmd, check=True)


def main():
    # Change to project root for consistent paths
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)

    print(f"Model: {MODEL}")
    print(f"Dataset: {DATASET}")
    print(f"Samples: {SAMPLES}")
    print(f"Max tokens: {MAX_TOKENS}")

    # 1. No CoT baseline
    run_eval(
        constraint="no_cot",
        name=f"{MODEL_SHORT}_{DATASET}_no_cot",
    )

    # 2. Standard CoT (unconstrained)
    run_eval(
        constraint="unconstrained",
        name=f"{MODEL_SHORT}_{DATASET}_cot",
    )

    # 3. Prompt repeats (with no_cot)
    for repeats in [2, 3, 4, 5, 10]:
        run_eval(
            constraint="no_cot",
            name=f"{MODEL_SHORT}_{DATASET}_repeat{repeats}",
            repeat_input=repeats,
        )

    # 4. Filler tokens (with no_cot)
    for filler in [10, 100, 1000]:
        run_eval(
            constraint="no_cot",
            name=f"{MODEL_SHORT}_{DATASET}_filler{filler}",
            filler_tokens=filler,
        )

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == "__main__":
    main()
