"""
Evaluation runner for the constrained reasoning framework.

This module contains the core run_eval function and system prompts.
"""

import argparse
from dotenv import load_dotenv
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval as inspect_eval, Epochs
from inspect_ai.model import ChatMessageUser, ChatMessageSystem
from inspect_ai.solver import generate, system_message, user_message, chain, assistant_message, solver, Solver, TaskState, Generate

from .datasets import load_dataset, get_scorer
from .constraints import get_constraint
from .datasets import DATASETS
from .constraints import CONSTRAINTS


load_dotenv()
# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


def short_model_name(model: str) -> str:
    """Extract short model name from full path (e.g., 'openrouter/openai/gpt-4o-mini' -> 'gpt-4o-mini')."""
    return model.split("/")[-1]


# ============ System Prompts ============

BASE_REASONING_PROMPT = """
Think about how you would solve the following problem. Reason step-by-step and output an explanation of how you would solve it.
Do not output the answer.
"""

BASE_ANSWER_WITH_REASONING_PROMPT = """
Now, please answer the question immediately, without showing reasoning. Your answer should be in the form "ANSWER:$ANSWER" (without quotes), where your answer is a single number.
"""

# ============ Solvers ====================


@solver
def insert_system_message(content: str, insert_at_beginning=True, **params: Any) -> Solver:
    """Solver which inserts a user message at the beginning of the conversation.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        kwargs = state.metadata | state.store._data | params
        if insert_at_beginning:
            state.messages.insert(
                0,
                ChatMessageSystem(
                    content=content, model=state.model.name
                )
            )
        else:
            state.messages.append(ChatMessageSystem(content=content, model=state.model.name))
        return state

    return solve

# ============ Evaluation Runner ============

def build_task(
    constraint_name: str | None,
    dataset_name: str,
    seed: int = 42,
    epochs: int = 1,
    name: str | None = None,
) -> Task:
    """Build a single evaluation task.

    Args:
        constraint_name: Reasoning constraint to apply (e.g., "unconstrained", "no_cot")
        dataset_name: Dataset to use
        seed: Random seed for reproducibility
        epochs: Number of times to run each sample (reduces variance via majority vote)
        name: Name for this task. Defaults to '{constraint}_{dataset}'

    Returns:
        Inspect Task
    """
    dataset = load_dataset(dataset_name, shuffle=True, seed=seed)
    scorer_fn = get_scorer(dataset_name)
    reasoning_prompt = BASE_REASONING_PROMPT
    if constraint_name:
        constraint = get_constraint(constraint_name)
        reasoning_prompt += constraint.system_prompt
    # Default name (CLI adds _repeatN suffix if needed)
    task_name = name or f"2stage_{constraint_name}_{dataset_name}"

    # Base solvers
    solvers = [
        insert_system_message(reasoning_prompt),
        generate(),
        insert_system_message(BASE_ANSWER_WITH_REASONING_PROMPT, insert_at_beginning=False),
        #assistant_message("ANSWER:"),
        generate(),
    ]

    return Task(
        name=task_name,
        dataset=dataset,
        solver=chain(*solvers),
        scorer=scorer_fn(),
        epochs=Epochs(epochs, "mode") if epochs > 1 else None,
    )


def run_eval(
    constraint_name: str,
    model: str = "openrouter/openai/gpt-4o-mini",
    dataset_name: str = "gsm8k",
    n_samples: int | None = 10,
    seed: int = 42,
    epochs: int = 1,
    name: str | None = None,
):
    """Run an evaluation with a specified constraint.

    Args:
        model: Model to evaluate (OpenRouter format, e.g., "openrouter/openai/gpt-4o-mini")
        dataset_name: Dataset to use
        constraint_name: Reasoning constraint to apply (e.g., "baseline", "no_cot")
        n_samples: Number of samples to evaluate (None = full dataset)
        seed: Random seed for reproducibility
        epochs: Number of times to run each sample (reduces variance via majority vote)
        name: Name for eval run. Defaults to '{constraint}_{dataset}'

    Returns:
        Inspect eval results
    """
    task = build_task(
        constraint_name=constraint_name,
        dataset_name=dataset_name,
        seed=seed,
        epochs=epochs,
        name=name,
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

def main():
    parser = argparse.ArgumentParser(
        description="Run constrained reasoning evaluations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m", "--model",
        default="openrouter/openai/gpt-4o-mini",
        help="Model to evaluate (OpenRouter format)",
    )
    parser.add_argument(
        "-d", "--dataset",
        default="gsm8k",
        choices=list(DATASETS.keys()),
        help="Dataset to use",
    )
    parser.add_argument(
        "-c", "--constraint",
        required=False,
        choices=list(CONSTRAINTS.keys()),
        help="Reasoning constraint to apply",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name for this eval run (shows in logs). Defaults to '{constraint}_{dataset}'",
    )
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=10,
        help="Number of samples from the dataset to evaluate (0 = full dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of times to run each sample (reduces variance via majority vote)",
    )

    args = parser.parse_args()

    # Build eval name
    eval_name = args.name or f"{args.constraint}_{args.dataset}_{short_model_name(args.model)}"

    print("Running evaluation...")
    print(f"  Name:       {eval_name}")
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Constraint: {args.constraint}")
    print(f"  Samples:    {args.n_samples}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Seed:       {args.seed}")
    print()

    results = run_eval(
        constraint_name=args.constraint,
        model=args.model,
        dataset_name=args.dataset,
        n_samples=args.n_samples if args.n_samples > 0 else None,
        seed=args.seed,
        epochs=args.epochs,
        name=eval_name,
    )

    return results

if __name__ == "__main__":
    main()