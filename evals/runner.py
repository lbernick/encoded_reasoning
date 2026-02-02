"""
Evaluation runner for the constrained reasoning framework.

This module contains the core run_eval function and system prompts.
"""

from pathlib import Path

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.solver import generate, system_message, chain

from .datasets import load_dataset, get_scorer
from .constraints import get_constraint


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


# ============ System Prompts ============

BASE_SYSTEM_PROMPT_COT = """
Solve the following problem. Show your reasoning in <reasoning> tags, then give your final answer in <answer> tags.

Example:
<reasoning>
Step-by-step work here...
</reasoning>
<answer>42</answer>
"""

BASE_SYSTEM_PROMPT_NO_COT = """
Solve the following problem. Give your final answer in <answer> tags.

Example:
<answer>42</answer>
"""


# ============ Evaluation Runner ============

def run_eval(
    constraint_name: str,
    model: str = "openrouter/openai/gpt-4o-mini",
    dataset_name: str = "gsm8k",
    n_samples: int = 10,
    seed: int = 42,
    name: str | None = None,
):
    """Run an evaluation with a specified constraint.

    Args:
        model: Model to evaluate (OpenRouter format, e.g., "openrouter/openai/gpt-4o-mini")
        dataset_name: Dataset to use
        constraint_name: Reasoning constraint to apply (e.g., "baseline", "no_cot")
        n_samples: Number of samples from the dataset to evaluate (not repeated runs)
        seed: Random seed for reproducibility
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

    task = Task(
        name=task_name,
        dataset=dataset,
        solver=chain(
            system_message(full_prompt),
            generate(),
        ),
        scorer=scorer_fn(),
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
