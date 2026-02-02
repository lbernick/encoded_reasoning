"""
Evaluation runner for the constrained reasoning framework.

This module contains the core run_eval function and system prompts.
"""

from pathlib import Path

from inspect_ai import Task, eval as inspect_eval, Epochs
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, system_message, chain, assistant_message, solver, TaskState

from .datasets import load_dataset, get_scorer
from .constraints import get_constraint


@solver
def repeat_input_solver(n: int):
    """Solver that repeats the user input n times."""
    async def solve(state: TaskState, generate):
        if n > 1 and state.messages:
            # Find the user message and repeat its content
            for msg in state.messages:
                if msg.role == "user":
                    original = msg.content
                    msg.content = "\n\n".join([original] * n)
                    break
        return state
    return solve


@solver
def filler_tokens_solver(n: int):
    """Solver that adds n filler tokens (periods) to the user prompt."""
    async def solve(state: TaskState, generate):
        if n > 0 and state.messages:
            # Find the user message and add filler
            for msg in state.messages:
                if msg.role == "user":
                    msg.content = msg.content + "\n" + "." * n
                    break
        return state
    return solve


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

def build_task(
    constraint_name: str,
    dataset_name: str,
    seed: int = 42,
    epochs: int = 1,
    repeat_input: int = 1,
    filler_tokens: int = 0,
    name: str | None = None,
) -> Task:
    """Build a single evaluation task.

    Args:
        constraint_name: Reasoning constraint to apply (e.g., "baseline", "no_cot")
        dataset_name: Dataset to use
        seed: Random seed for reproducibility
        epochs: Number of times to run each sample (reduces variance via majority vote)
        repeat_input: Number of times to repeat the question in the prompt
        filler_tokens: Number of filler tokens (periods) to add to the prompt
        name: Name for this task. Defaults to '{constraint}_{dataset}'

    Returns:
        Inspect Task
    """
    dataset = load_dataset(dataset_name, shuffle=True, seed=seed)
    scorer_fn = get_scorer(dataset_name)
    constraint = get_constraint(constraint_name)

    # Choose base prompt based on whether constraint expects reasoning
        base_prompt = BASE_SYSTEM_PROMPT_COT if constraint.expects_reasoning else BASE_SYSTEM_PROMPT_NO_COT

        full_prompt = base_prompt + "\n" + constraint.system_prompt
    # Default name (CLI adds _repeatN suffix if needed)
    task_name = name or f"{constraint_name}_{dataset_name}"

    # Build solver chain - add assistant prefill for no-COT to prevent reasoning
    # Stop generation after </answer> to save tokens
    gen_config = GenerateConfig(stop_seqs=["</answer>"])

    # Base solvers
    solvers = [system_message(full_prompt)]

    # Optionally repeat the input n times
    if repeat_input > 1:
        solvers.append(repeat_input_solver(repeat_input))

    # Optionally add filler tokens
    if filler_tokens > 0:
        solvers.append(filler_tokens_solver(filler_tokens))

    # Add prefill for no-COT, then generate
    if not constraint.expects_reasoning:
        solvers.append(assistant_message("<answer>"))

    solvers.append(generate(config=gen_config))

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
    repeat_input: int = 1,
    filler_tokens: int = 0,
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
        repeat_input: Number of times to repeat the question in the prompt
        filler_tokens: Number of filler tokens (periods) to add to the prompt
        name: Name for eval run. Defaults to '{constraint}_{dataset}'

    Returns:
        Inspect eval results
    """
    task = build_task(
        constraint_name=constraint_name,
        dataset_name=dataset_name,
        seed=seed,
        epochs=epochs,
        repeat_input=repeat_input,
        filler_tokens=filler_tokens,
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
