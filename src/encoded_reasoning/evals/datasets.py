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
from dotenv import load_dotenv
from enum import Enum

import random
import re
from typing import Any, Callable

from pathlib import Path
import json as json_module
import urllib.request

import pandas as pd

from inspect_ai.dataset import Sample, hf_dataset, Dataset, MemoryDataset
from inspect_ai.scorer import Scorer, Score, Target, accuracy, stderr, scorer
from inspect_ai.solver import TaskState

load_dotenv()
# Letters for multiple choice options
ANSWER_LETTERS = ["A", "B", "C", "D"]


# ============ GSM8K ============


def gsm8k_record_to_sample(record: dict) -> Sample:
    """Convert GSM8K record to Sample, extracting numeric answer after ####."""
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", record["answer"])
    target = match.group(1).replace(",", "") if match else record["answer"]
    return Sample(input=record["question"], target=target)

def gsm8k_format_func(example):
    question = example["question"]
    answer = extract_answer_from_gsm8k(example["answer"])
    return question, answer

def extract_answer_from_gsm8k(answer_text: str) -> str:
    """
    Extract the final numerical answer from GSM8K format.
    GSM8K answers are in the format: "Step by step explanation\n#### 42"
    """
    # GSM8K format: answer is after ####
    if "####" in answer_text:
        answer = answer_text.split("####")[-1].strip()
        # Remove commas from numbers (e.g., "1,000" -> "1000")
        answer = answer.replace(",", "")
        return answer
    return answer_text.strip()


def extract_number_answer(text: str) -> str | None:
    """Extract numeric answer from model output."""
    if "ANSWER:" in text:
        return text.split("ANSWER:")[1].strip()
    # <answer>X</answer> pattern (full tags, COT case)
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip().replace(",", "")

    # X</answer> pattern (prefilled, no-COT case)
    match = re.search(r"^(-?\d+(?:,\d+)*(?:\.\d+)?)\s*</answer>", text.strip())
    if match:
        return match.group(1).replace(",", "")

    return None


@scorer(metrics=[accuracy(), stderr()])
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


@scorer(metrics=[accuracy(), stderr()])
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


# ============ MAWPS ============


def mawps_record_to_sample(record: dict) -> Sample:
    """Convert MAWPS record to Sample.

    MAWPS (MAth Word ProblemS) contains elementary math word problems
    with arithmetic operations (addition, subtraction, multiplication, division).
    """
    # Answer is a list, extract the first element
    answer = record["answer"]
    if isinstance(answer, list):
        answer = answer[0]

    # Normalize answer: remove trailing .0 for whole numbers
    answer_str = str(answer)
    if answer_str.endswith(".0"):
        answer_str = answer_str[:-2]

    return Sample(
        input=record["question"],
        target=answer_str,
        metadata={
            "id": record.get("id", ""),
            "type": record.get("type", ""),
            "cot": record.get("cot", []),
        },
    )

def mawps_format_func(example):
    return example["question"], example["answer"][0]

@scorer(metrics=[accuracy(), stderr()])
def mawps_scorer() -> Scorer:
    """Scorer for MAWPS - extracts numeric answer and compares."""

    async def score(state: TaskState, target: Target) -> Score:
        predicted = extract_number_answer(state.output.completion)

        if predicted is None:
            return Score(
                value=False,
                answer=None,
                explanation=f"Could not extract answer. Expected: {target.text}",
            )

        # Normalize both for comparison
        # Remove trailing .0 from predicted if present
        pred_normalized = predicted
        if pred_normalized.endswith(".0"):
            pred_normalized = pred_normalized[:-2]

        target_normalized = target.text
        if target_normalized.endswith(".0"):
            target_normalized = target_normalized[:-2]

        return Score(
            value=(pred_normalized == target_normalized),
            answer=predicted,
            explanation=f"Predicted: {predicted}, Expected: {target.text}",
        )

    return score


def mawps2_record_to_sample(record: dict) -> Sample:
    # Answer is a list, extract the first element
    answer = extract_answer_from_gsm8k(record["answer"])

    return Sample(
        input=record["question"],
        target=answer,
        metadata={
            "id": record.get("id", ""),
            "type": record.get("type", ""),
            "cot": record.get("cot", []),
        },
    )

def mawps2_format_func(example):
    return example["question"], extract_answer_from_gsm8k(example["answer"])

# ============ MoreHopQA ============

# MoreHopQA data sources
MOREHOPQA_PARQUET_URLS = {
    "verified": "https://huggingface.co/datasets/alabnii/morehopqa/resolve/main/verified/test-00000-of-00001.parquet",
}
MOREHOPQA_JSON_URL = "https://huggingface.co/datasets/alabnii/morehopqa/raw/main/data/with_human_verification.json"

# Path to list of question IDs that Opus got correct
MOREHOPQA_OPUS_CORRECT_IDS_PATH = (
    Path(__file__).parent / "data" / "morehopqa_opus_correct_ids.json"
)


def morehopqa_record_to_sample(row: dict) -> Sample:
    """Convert MoreHopQA record to Sample.

    MoreHopQA is a multi-hop QA dataset requiring generative answers.
    Questions require multiple reasoning steps including factual, commonsense,
    arithmetic, and symbolic reasoning.
    """
    return Sample(
        input=row["question"],
        target=str(row["answer"]),
        metadata={
            "question_id": row.get("_id", ""),
            "reasoning_type": row.get("reasoning_type", ""),
            "no_of_hops": row.get("no_of_hops", ""),
            "answer_type": row.get("answer_type", ""),
            "previous_question": row.get("previous_question", ""),
            "previous_answer": row.get("previous_answer", ""),
        },
    )


def extract_text_answer(text: str) -> str | None:
    """Extract text answer from model output."""
    # <answer>X</answer> pattern (full tags, COT case)
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # X</answer> pattern (prefilled, no-COT case)
    match = re.search(r"^(.*?)</answer>", text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (lowercase, strip whitespace/punctuation)."""
    answer = answer.lower().strip()
    # Remove trailing punctuation
    answer = re.sub(r"[.!?]+$", "", answer)
    # Normalize whitespace
    answer = re.sub(r"\s+", " ", answer)
    return answer


@scorer(metrics=[accuracy(), stderr()])
def morehopqa_scorer() -> Scorer:
    """Scorer for MoreHopQA - extracts text answer and does normalized comparison."""

    async def score(state: TaskState, target: Target) -> Score:
        predicted = extract_text_answer(state.output.completion)

        if predicted is None:
            return Score(
                value=False,
                answer=None,
                explanation=f"Could not extract answer. Expected: {target.text}",
            )

        # Normalize both answers for comparison
        pred_normalized = normalize_answer(predicted)
        target_normalized = normalize_answer(target.text)

        is_correct = pred_normalized == target_normalized

        return Score(
            value=is_correct,
            answer=predicted,
            explanation=f"Predicted: {predicted}, Expected: {target.text}",
        )

    return score

# ============ BOOLQ ============

def boolq_record_to_sample(record: dict) -> Sample:
    return Sample(input=record["question"], target=str(record["answer"]))

def boolq_format_func(example):
    question = example["question"]
    answer = example["answer"]
    return question, answer

def extract_boolean_answer(text: str) -> str | None:
    if text.lower() == "true" or text.lower() == "false":
        return text
    if "ANSWER:" in text:
        return text.split("ANSWER:")[1].strip()
    # <answer>X</answer> pattern (full tags, COT case)
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Look for boolean strings (case-insensitive)
        bool_match = re.search(r"\b(true|false)\b", content, re.IGNORECASE)
        if bool_match:
            return bool_match.group(1)
        return content

    # X</answer> pattern (prefilled, no-COT case) - look for boolean strings
    match = re.search(r"^(true|false)\s*</answer>", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1)

    return None

@scorer(metrics=[accuracy(), stderr()])
def boolq_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        predicted = extract_boolean_answer(state.output.completion)

        if predicted is None:
            return Score(
                value=False,
                answer=None,
                explanation=f"Could not extract answer. Expected: {target.text}",
            )

        return Score(
            value=(predicted.lower() == target.text.lower()),
            answer=predicted,
            explanation=f"Predicted: {predicted}, Expected: {target.text}",
        )

    return score


# ============ Dataset Registry ============

DatasetRecipe = dict[str, Any]

class DatasetType(Enum):
    MATHEMATICAL = 1
    MCQ = 2
    FREE_RESPONSE = 3
    BOOL = 4

DATASETS: dict[str, DatasetRecipe] = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_name": "main",
        "train_split": "train",
        "test_split": "test",
        "config": "main",
        "record_to_sample": gsm8k_record_to_sample,
        "format_func": gsm8k_format_func,
        "scorer": gsm8k_scorer,
        "type": DatasetType.MATHEMATICAL,
    },
    "gpqa": {
        "hf_path": "Idavidrein/gpqa",
        "hf_name": "gpqa_diamond",  # Hardest subset, 198 questions
        "train_split": "train",
        "test_split": "train",  # GPQA only has train split
        "record_to_sample": gpqa_record_to_sample,
        "scorer": gpqa_scorer,
        "type": DatasetType.MCQ,
    },
    "morehopqa": {
        # Uses custom loader (HF loading script is deprecated)
        # Human-verified subset with 1118 samples
        "json_url": MOREHOPQA_JSON_URL,
        "record_to_sample": morehopqa_record_to_sample,
        "scorer": morehopqa_scorer,
        "system_prompt": "If the answer is a date, format it as YYYY-MM-DD.",
        "type": DatasetType.FREE_RESPONSE,
    },
    "morehopqa_opus_correct": {
        # Subset of MoreHopQA that Opus 4.5 answered correctly (606 samples)
        # Useful for testing other models on questions a strong model can solve
        "json_url": MOREHOPQA_JSON_URL,
        "record_to_sample": morehopqa_record_to_sample,
        "scorer": morehopqa_scorer,
        "system_prompt": "If the answer is a date, format it as YYYY-MM-DD.",
        "filter_ids_file": MOREHOPQA_OPUS_CORRECT_IDS_PATH,
        "type": DatasetType.FREE_RESPONSE,
    },
    "mawps": {
        # MAWPS: elementary math word problems (1921 samples)
        # Uses nguyen-brat/mawps which has actual numbers in questions
        "hf_path": "nguyen-brat/mawps",
        "train_split": "train",
        "test_split": "train",  # Only has train split
        "config": "default",
        "record_to_sample": mawps_record_to_sample,
        "format_func": mawps_format_func,
        "scorer": mawps_scorer,
        "type": DatasetType.MATHEMATICAL,
    },
    "mawps2": {
        "hf_path": "garrethlee/MAWPS",
        "train_split": "train", 
        "test_split": "test",
        "config": "default",
        "record_to_sample": mawps2_record_to_sample,
        "format_func": mawps2_format_func,
        "scorer": mawps_scorer,
        "type": DatasetType.MATHEMATICAL,
    },
    "boolq": {
        "hf_path": "google/boolq",
        "train_split": "train", 
        "test_split": "validation",
        "record_to_sample": boolq_record_to_sample,
        "format_func": boolq_format_func,
        "scorer": boolq_scorer,
        "type": DatasetType.BOOL,
    },
}


def load_dataset(
    name: str,
    split: str | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Dataset:
    """Load a dataset by name using Inspect's hf_dataset or custom loader.

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

    # Handle datasets that need custom loading (e.g., deprecated HF loading scripts)
    if "json_url" in recipe:
        # Load from JSON URL
        json_url = recipe["json_url"]
        with urllib.request.urlopen(json_url) as response:
            data = json_module.loads(response.read().decode())

        # Filter by IDs if specified
        filter_ids_file = recipe.get("filter_ids_file")
        if filter_ids_file and Path(filter_ids_file).exists():
            with open(filter_ids_file) as f:
                allowed_ids = set(json_module.load(f))
            data = [row for row in data if row.get("_id") in allowed_ids]

        record_to_sample = recipe["record_to_sample"]
        samples = [record_to_sample(row) for row in data]

        if shuffle:
            random.seed(seed)
            random.shuffle(samples)

        return MemoryDataset(samples)

    if "parquet_url" in recipe:
        df = pd.read_parquet(recipe["parquet_url"])
        record_to_sample = recipe["record_to_sample"]
        samples = [record_to_sample(df.iloc[i].to_dict()) for i in range(len(df))]

        if shuffle:
            random.seed(seed)
            random.shuffle(samples)

        return MemoryDataset(samples)

    # Standard HuggingFace dataset loading
    return hf_dataset(
        recipe["hf_path"],
        name=recipe.get("hf_name"),
        split=split or recipe["test_split"],
        sample_fields=recipe["record_to_sample"],
        shuffle=shuffle,
        seed=seed,
    )


def get_scorer(name: str) -> Callable[[], Scorer]:
    """Get the scorer function for a dataset."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name]["scorer"]


def get_dataset_system_prompt(name: str) -> str | None:
    """Get the dataset-specific system prompt, if any."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name].get("system_prompt")

def get_dataset_type(name: str) -> DatasetType | None:
    """Get the dataset-specific system prompt, if any."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name].get("type")
