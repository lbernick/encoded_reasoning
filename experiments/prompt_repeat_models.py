#!/usr/bin/env python3
"""
Prompt repeat ablation across multiple models.

Tests how different models respond to repeated prompts (no CoT).
Includes Qwen and Gemma model families.
"""

import subprocess
import sys
from pathlib import Path

# Configuration
SAMPLES = 300
DATASET = "gsm8k"
MAX_TOKENS = 512
LOG_DIR = "gpt52"

# Repeat counts to test
REPEATS = [1, 2, 3, 5, 10]

# Models to test (openrouter format)
MODELS: dict[str, str] = {
    # Qwen 2.5 series
    # "qwen2.5-7b": "openrouter/qwen/qwen-2.5-7b-instruct",
    "qwen2.5-14b": "openrouter/qwen/qwen-2.5-14b",
    "qwen2.5-32b": "openrouter/qwen/qwen-2.5-32b",
    # "qwen2.5-72b": "openrouter/qwen/qwen-2.5-72b-instruct",
    # Gemma 3 series
    "gemma3-1b": "openrouter/google/gemma-3-1b-it",
    "gemma3-4b": "openrouter/google/gemma-3-4b-it",
    "gemma3-12b": "openrouter/google/gemma-3-12b-it",
    "gemma3-27b": "openrouter/google/gemma-3-27b-it",
}


def run_eval(
    model_id: str,
    model_short: str,
    constraint: str = "no_cot",
    repeat_input: int = 1,
) -> None:
    """Run a single evaluation."""
    if constraint == "unconstrained":
        name = f"{model_short}_{DATASET}_cot"
    else:
        name = f"{model_short}_{DATASET}_repeat{repeat_input}"
    
    cmd = [
        sys.executable, "-m", "encoded_reasoning",
        "--model", model_id,
        "--dataset", DATASET,
        "--constraint", constraint,
        "--name", name,
        "-n", str(SAMPLES),
        "--max-tokens", str(MAX_TOKENS),
        "--log-dir", LOG_DIR,
    ]

    if repeat_input > 1:
        cmd.extend(["--repeat-input", str(repeat_input)])

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Model: {model_id}")
    print(f"Repeat: {repeat_input}")
    print(f"{'='*60}\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {name} failed with exit code {e.returncode}")


def main():
    # Change to project root for consistent paths
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)

    print("Prompt Repeat Ablation Experiment")
    print(f"Dataset: {DATASET}")
    print(f"Samples: {SAMPLES}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Repeats: {REPEATS}")
    print(f"Models: {list(MODELS.keys())}")
    print()

    # +1 for CoT baseline per model
    total_runs = len(MODELS) * (len(REPEATS) + 1)
    current_run = 0

    for model_short, model_id in MODELS.items():
        # CoT baseline
        current_run += 1
        print(f"\n[{current_run}/{total_runs}] ", end="")
        run_eval(
            model_id=model_id,
            model_short=model_short,
            constraint="unconstrained",
        )

        # No-CoT with repeats
        for repeat in REPEATS:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] ", end="")
            run_eval(
                model_id=model_id,
                model_short=model_short,
                constraint="no_cot",
                repeat_input=repeat,
            )

    print("\n" + "="*60)
    print("All experiments completed!")
    print(f"Total runs: {total_runs}")
    print("="*60)


if __name__ == "__main__":
    main()
