"""
Dataset recipes for the constrained reasoning evaluation framework.

Each dataset defines:
- hf_path: HuggingFace dataset path
- hf_name: HuggingFace dataset subset name (optional)
- split: Default split to use
- record_to_sample: Function to convert a record to an Inspect Sample
- scorer: Scorer function for evaluating model outputs
"""

# Load environment variables BEFORE any other imports
# (Inspect reads OPENROUTER_API_KEY at import time)
import os
from dotenv import load_dotenv
load_dotenv()

import re
from typing import Any, Callable

from inspect_ai.dataset import Sample, hf_dataset, Dataset
from inspect_ai.scorer import Scorer, Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState


# ============ GSM8K ============

def gsm8k_record_to_sample(record: dict) -> Sample:
    """Convert GSM8K record to Sample, extracting numeric answer after ####."""
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", record["answer"])
    target = match.group(1).replace(",", "") if match else record["answer"]
    return Sample(input=record["question"], target=target)


@scorer(metrics=[accuracy()])
def gsm8k_scorer() -> Scorer:
    """Scorer for GSM8K that extracts answers from <answer> tags or last number."""

    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion

        # Try <answer> tags first
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if answer_match:
            predicted = answer_match.group(1).strip()
            # Clean up any commas in numbers
            predicted = predicted.replace(",", "")
        else:
            # Fallback: extract last number from response
            numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
            predicted = numbers[-1].replace(",", "") if numbers else None

        if predicted is None:
            return Score(
                value=False,
                answer=None,
                explanation=f"Could not extract answer. Expected: {target.text}",
            )

        correct = predicted == target.text
        return Score(
            value=correct,
            answer=predicted,
            explanation=f"Predicted: {predicted}, Expected: {target.text}",
        )

    return score


# ============ Dataset Registry ============

DatasetRecipe = dict[str, Any]

DATASETS: dict[str, DatasetRecipe] = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "split": "test",
        "record_to_sample": gsm8k_record_to_sample,
        "scorer": gsm8k_scorer,
    },
}


def load_dataset(
    name: str,
    split: str | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Dataset:
    """Load a dataset by name using Inspect's hf_dataset.

    Args:
        name: Dataset name (must be in DATASETS registry)
        split: Override the default split
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling

    Returns:
        Inspect Dataset ready for use in a Task
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    recipe = DATASETS[name]
    return hf_dataset(
        recipe["hf_path"],
        name=recipe.get("hf_name"),
        split=split or recipe["split"],
        sample_fields=recipe["record_to_sample"],
        shuffle=shuffle,
        seed=seed,
    )


def get_scorer(name: str) -> Callable[[], Scorer]:
    """Get the scorer function for a dataset."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name]["scorer"]
