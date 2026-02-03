"""
Evaluation runner for the constrained reasoning framework.

This module contains the core run_eval function and system prompts.
"""

from pathlib import Path

from inspect_ai import Task, eval as inspect_eval, Epochs

from inspect_ai.solver import generate, system_message, chain, assistant_message, solver, TaskState

from inspect_ai.model import get_model, GenerateConfig

from .datasets import load_dataset, get_scorer
from .constraints import get_constraint
import logit_masking.model_api  # noqa: F401 — registers hf-masked provider


@solver
def repeat_input_solver(n: int):
    """Solver that repeats the user input n times."""
    async def solve(state: TaskState, generate):
        if n > 1 and state.messages:
            # Create new message list to avoid mutating shared message objects
            # (which can cause all samples to have the same question)
            from copy import copy
            new_messages = []
            for msg in state.messages:
                if msg.role == "user" and isinstance(msg.content, str):
                    # Create a copy with repeated content instead of mutating original
                    new_msg = copy(msg)
                    new_msg.content = "\n\n".join([msg.content] * n)
                    new_messages.append(new_msg)
                else:
                    new_messages.append(msg)
            state.messages = new_messages
        return state
    return solve


@solver
def filler_tokens_solver(n: int):
    """Solver that adds n filler tokens (periods) to the user prompt."""
    async def solve(state: TaskState, generate):
        if n > 0 and state.messages:
            # Create new message list to avoid mutating shared message objects
            # (which can cause all samples to have the same question)
            from copy import copy
            new_messages = []
            for msg in state.messages:
                if msg.role == "user" and isinstance(msg.content, str):
                    # Create a copy with filler content instead of mutating original
                    new_msg = copy(msg)
                    new_msg.content = msg.content + "\n" + "." * n
                    new_messages.append(new_msg)
                else:
                    new_messages.append(msg)
            state.messages = new_messages
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
        constraint_name: Reasoning constraint to apply (e.g., "unconstrained", "no_cot")
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
    task_name = name or f"{constraint_name}_{dataset_name}"

    solvers = [system_message(full_prompt)]

    if repeat_input > 1:
        solvers.append(repeat_input_solver(repeat_input))

    if filler_tokens > 0:
        solvers.append(filler_tokens_solver(filler_tokens))

    if not constraint.expects_reasoning:
        solvers.append(assistant_message("<answer>"))

    solvers.append(generate())

    return Task(
        name=task_name,
        dataset=dataset,
        solver=chain(*solvers),
        scorer=scorer_fn(),
        epochs=Epochs(epochs, "mode") if epochs > 1 else None,
    )


def _resolve_model(model: str, constraint_name: str):
    """Resolve model string to a Model instance when logit masking is needed.

    If the model uses the ``hf/`` prefix and the constraint defines an
    ``allowed_token_filter``, returns a ``Model`` instance backed by the
    ``hf-masked`` provider. Otherwise returns the original string and lets
    inspect resolve it normally.
    """
    if model.startswith("hf/"):
        constraint = get_constraint(constraint_name)
        if constraint.allowed_token_filter is not None:
            hf_model_name = model.removeprefix("hf/")
            return get_model(
                f"hf-masked/{hf_model_name}",
                allowed_token_filter=constraint.allowed_token_filter,
            )
    return model


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
    max_tokens: int | None = None,
):
    """Run an evaluation with a specified constraint.

    Args:
        model: Model identifier. Use "hf/model-name" for local HuggingFace models,
            or an OpenRouter path (e.g., "openrouter/openai/gpt-4o-mini") for API models.
            When a hf/ model is used and the constraint defines allowed_token_filter,
            logit masking is applied automatically via the hf-masked provider.
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

    resolved_model = _resolve_model(model, constraint_name)

    eval_kwargs = {}
    if max_tokens is not None:
        eval_kwargs["config"] = GenerateConfig(max_tokens=max_tokens)

    results = inspect_eval(
        task,
        model=resolved_model,
        limit=n_samples,
        log_dir=str(LOG_DIR),
        metadata={
            "constraint": constraint_name,
            "dataset": dataset_name,
            "seed": seed,
        },
        **eval_kwargs,
    )

    return results
