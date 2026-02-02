"""
Evaluation runner for the constrained reasoning framework.

This module contains the core run_eval function and system prompts.
"""

from pathlib import Path

from inspect_ai import Task, eval as inspect_eval, Epochs
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, system_message, chain, assistant_message

from .datasets import load_dataset, get_scorer
from .constraints import get_constraint


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


# ============ System Prompts ============

BASE_SYSTEM_PROMPT_COT = """
Solve the following problem. Show your reasoning in <reasoning> tags, then give your final answer in <answer> tags.

For multiple choice questions, answer with just the letter (A, B, C, or D).
For math problems, answer with just the number.

Example (multiple choice):
<reasoning>
Step-by-step work here...
</reasoning>
<answer>B</answer>

Example (math):
<reasoning>
Step-by-step work here...
</reasoning>
<answer>42</answer>
"""

BASE_SYSTEM_PROMPT_NO_COT = """
Solve the following problem. Give your final answer in <answer> tags.

For multiple choice questions, answer with just the letter (A, B, C, or D).
For math problems, answer with just the number.

Example:
<answer>B</answer>
"""


# ============ Evaluation Runner ============

def run_eval(
    constraint_name: str,
    model: str = "openrouter/openai/gpt-4o-mini",
    dataset_name: str = "gsm8k",
    n_samples: int = 10,
    seed: int = 42,
    epochs: int = 1,
    name: str | None = None,
):
    """Run an evaluation with a specified constraint.

    Args:
        model: Model to evaluate (OpenRouter format, e.g., "openrouter/openai/gpt-4o-mini")
        dataset_name: Dataset to use
        constraint_name: Reasoning constraint to apply (e.g., "baseline", "no_cot")
        n_samples: Number of samples from the dataset to evaluate (not repeated runs)
        seed: Random seed for reproducibility
        epochs: Number of times to run each sample (reduces variance via majority vote)
        name: Name for this eval run (shows in logs). Defaults to '{constraint}_{dataset}'

    Returns:
        Inspect eval results
    """
    dataset = load_dataset(dataset_name, shuffle=True, seed=seed)
    scorer_fn = get_scorer(dataset_name)
    constraint = get_constraint(constraint_name)

    # Choose base prompt based on whether constraint expects reasoning
    base_prompt = BASE_SYSTEM_PROMPT_COT if constraint.expects_reasoning else BASE_SYSTEM_PROMPT_NO_COT
    full_prompt = base_prompt + "\n" + constraint.system_prompt

    # Default name if not provided
    task_name = name or f"{constraint_name}_{dataset_name}"

    # Build solver chain - add assistant prefill for no-COT to prevent reasoning
    # Stop generation after </answer> to save tokens
    gen_config = GenerateConfig(stop_seqs=["</answer>"])

    if constraint.expects_reasoning:
        solver = chain(
            system_message(full_prompt),
            generate(config=gen_config),
        )
    else:
        # Prefill with "<answer>" to force immediate answer without reasoning
        solver = chain(
            system_message(full_prompt),
            assistant_message("<answer>"),
            generate(config=gen_config),
        )

    task = Task(
        name=task_name,
        dataset=dataset,
        solver=solver,
        scorer=scorer_fn(),
        epochs=Epochs(epochs, "mode") if epochs > 1 else None,
    )

    results = inspect_eval(
        task,
        model=model,
        limit=n_samples,
        log_dir=str(LOG_DIR),
        metadata={
            "constraint": constraint_name,
            "dataset": dataset_name,
            "seed": seed,
        },
    )

    return results
