from dataclasses import dataclass
from typing import Callable


@dataclass
class DatasetRecipe:
    name: str
    format_func: Callable
    config: str = None
    split: str = None
    data_files: str = None


# ============================================================================
# GSM8K
# ============================================================================


def format_gsm8k(example):
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


GSM8K = DatasetRecipe(
    name="gsm8k",
    config="main",
    split="train",
    format_func=format_gsm8k,
)

# ============================================================================
# AddSub
# ============================================================================


def format_addsub(example):
    return example["input"], example["output_answer"]


ADDSUB = DatasetRecipe(
    name="allenai/lila",
    config="addsub",
    split="train",
    format_func=format_addsub,
)


def format_mawps(example):
    return example["question"], extract_answer_from_gsm8k(example["answer"])


MAWPS = DatasetRecipe(
    name="garrethlee/MAWPS",
    config="default",
    split="train",
    format_func=format_mawps,
)
# ============================================================================
# Simple Math
# ============================================================================


def format_simple(example):
    return example["question"], example["answer"]


SIMPLE_MATH = DatasetRecipe(
    name="simple_math",
    data_files="finetuning/data/simple_math_problems.json",
    format_func=format_simple,
)

# ============================================================================
# All datasets
# ============================================================================

DATASETS = {
    "gsm8k": GSM8K,
    "simple_math": SIMPLE_MATH,
    "mawps": MAWPS,
    # "addsub": ADDSUB
}
