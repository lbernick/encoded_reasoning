#!/usr/bin/env python3
"""
GPT-5.2 MoreHopQA experiments.

Tests three reasoning modes:
- CoT (unconstrained) - standard chain-of-thought
- No CoT (no_cot) - answer immediately without reasoning
- Only emojis (only_emojis) - reasoning restricted to emoji characters

Uses OpenRouter API.
"""

import subprocess
import sys
from pathlib import Path

# Configuration
MODEL = "openrouter/openai/gpt-5.2"
MODEL_SHORT = "gpt52"
SAMPLES = 100
DATASET = "morehopqa"
MAX_TOKENS = 512
LOG_DIR = "gpt52"

# Constraints to test
CONSTRAINTS = ["unconstrained", "no_cot", "only_emojis"]


def run_eval(constraint: str) -> None:
    """Run a single evaluation."""
    name = f"{MODEL_SHORT}-{DATASET}-{constraint}"
    
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
    print(f"Log dir: {LOG_DIR}")
    print(f"\nConstraints to test: {CONSTRAINTS}")

    for constraint in CONSTRAINTS:
        run_eval(constraint)

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == "__main__":
    main()
