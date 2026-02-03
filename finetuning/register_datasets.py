

from dataclasses import dataclass
from typing import Callable
from datasets import load_dataset
import random

@dataclass
class DatasetRecipe:
    name: str
    format_func: Callable
    config: str = None
    split: str = None
    data_files: str = None

def prepare_dataset(custom_dataset: DatasetRecipe, system_prompt: str, n_samples: int | None):
    """Load and format the dataset for training."""
    print(f"Loading {custom_dataset.name} dataset...")
    if custom_dataset.data_files:
        dataset = load_dataset("json", data_files=custom_dataset.data_files)["train"]
    else:
        dataset = load_dataset(custom_dataset.name, custom_dataset.config, split=custom_dataset.split)

    if n_samples is not None:
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        dataset = dataset.select(indices)
    
    print(f"Dataset loaded: {len(dataset)} examples")
    
    def format_example(example):
        """Format a single GSM8K example."""
        question, answer = custom_dataset.format_func(example)
        conversation =  [{"role": "system", "content": system_prompt},
        {"role": "user", "content": question}]
        
        return {
            "prompt": conversation,  # Store as conversation format
            "answer": answer,  # Store for reward function
        }
    
    formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    
    # Print a sample
    print("\n" + "="*80)
    print("SAMPLE FORMATTED EXAMPLE:")
    print("="*80)
    sample = formatted_dataset[0]
    print("Conversation:")
    for msg in sample['prompt']:
        print(f"  [{msg['role']}]: {msg['content']}")
    print(f"\nExpected answer: {sample['answer']}")
    print("="*80 + "\n")
    
    return formatted_dataset

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
}