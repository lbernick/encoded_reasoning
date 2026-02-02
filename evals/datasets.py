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

import random
import re
from typing import Any, Callable

from inspect_ai.dataset import Sample, hf_dataset, Dataset
from inspect_ai.scorer import Scorer, Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState

# Letters for multiple choice options
ANSWER_LETTERS = ["A", "B", "C", "D"]


# ============ GSM8K ============

def gsm8k_record_to_sample(record: dict) -> Sample:
    """Convert GSM8K record to Sample, extracting numeric answer after ####."""
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", record["answer"])
    target = match.group(1).replace(",", "") if match else record["answer"]
    return Sample(input=record["question"], target=target)


def extract_number_answer(text: str) -> str | None:
    """Extract numeric answer from model output."""
    # <answer>X</answer> pattern (full tags, COT case)
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip().replace(",", "")

    # X</answer> pattern (prefilled, no-COT case)
    match = re.search(r"^(-?\d+(?:,\d+)*(?:\.\d+)?)\s*</answer>", text.strip())
    if match:
        return match.group(1).replace(",", "")

    return None


@scorer(metrics=[accuracy()])
def gsm8k_scorer() -> Scorer:
    """Scorer for GSM8K that extracts answers from <answer> tags or last number."""

    async def score(state: TaskState, target: Target) -> Score:
        predicted = extract_number_answer(state.output.completion)

        if predicted is None:
            return Score(
                value=False,
                answer=None,
                explanation=f"Could not extract answer. Expected: {target.text}",
            )

        return Score(
            value=(predicted == target.text),
            answer=predicted,
            explanation=f"Predicted: {predicted}, Expected: {target.text}",
        )

    return score


# ============ GPQA ============

def gpqa_record_to_sample(record: dict) -> Sample:
    """Convert GPQA record to Sample with shuffled multiple choice options."""
    # Collect all choices
    choices = [
        record["Correct Answer"],
        record["Incorrect Answer 1"],
        record["Incorrect Answer 2"],
        record["Incorrect Answer 3"],
    ]
    # Clean up whitespace
    choices = [c.strip() for c in choices]

    # Shuffle choices (correct answer starts at index 0)
    indices = list(range(4))
    random.shuffle(indices)
    shuffled_choices = [choices[i] for i in indices]

    # Find where correct answer ended up
    correct_index = indices.index(0)
    target = ANSWER_LETTERS[correct_index]

    # Format question with choices included in the input
    question = record["Question"].strip()
    formatted_choices = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(ANSWER_LETTERS, shuffled_choices)
    )
    full_input = f"{question}\n\n{formatted_choices}"

    return Sample(
        input=full_input,
        choices=shuffled_choices,
        target=target,
        metadata={
            "domain": record.get("High-level domain", ""),
            "subdomain": record.get("Subdomain", ""),
        },
    )


def extract_choice_answer(text: str) -> str | None:
    """Extract multiple choice letter (A-D) from model output."""
    # <answer>X</answer> pattern (full tags, COT case)
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        answer = match.group(1).strip().upper()
        if answer and answer[0] in ANSWER_LETTERS:
            return answer[0]

    # X</answer> pattern (prefilled, no-COT case)
    match = re.search(r"^([A-D])\s*</answer>", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


@scorer(metrics=[accuracy()])
def gpqa_scorer() -> Scorer:
    """Scorer for GPQA multiple choice - extracts letter answer from <answer> tags."""

    async def score(state: TaskState, target: Target) -> Score:
        predicted = extract_choice_answer(state.output.completion)

        if predicted is None:
            return Score(
                value=False,
                answer=None,
                explanation=f"Could not extract valid answer (A-D). Expected: {target.text}",
            )

        return Score(
            value=(predicted == target.text),
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
    "gpqa": {
        "hf_path": "Idavidrein/gpqa",
        "hf_name": "gpqa_diamond",  # Hardest subset, 198 questions
        "split": "train",  # GPQA only has train split
        "record_to_sample": gpqa_record_to_sample,
        "scorer": gpqa_scorer,
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
