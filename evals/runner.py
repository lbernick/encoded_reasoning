"""
Evaluation runner for the constrained reasoning framework.

This module contains the core run_eval function and system prompts.
Supports both single-stage and two-stage evaluation modes.
"""

from pathlib import Path
from typing import Any

from inspect_ai import Task, eval as inspect_eval, Epochs
from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import (
    generate,
    system_message,
    chain,
    assistant_message,
    solver,
    Solver,
    TaskState,
    Generate,
)

from .datasets import load_dataset, get_scorer, get_dataset_system_prompt
from .constraints import get_constraint


from inspect_ai.model import get_model
import logit_masking.model_api  # noqa: F401 — registers hf-masked provider
# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


# ============ System Prompts ============

# Single-stage prompts
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
For free response problems, answer with just the final answer.

Example:
<answer>B</answer>
"""

# Two-stage prompts
BASE_REASONING_PROMPT = """
Think about how you would solve the following problem. Reason step-by-step and output an explanation of how you would solve it.
Do not output the answer.
"""

BASE_ANSWER_WITH_REASONING_PROMPT = """
Now, please answer the question immediately, without showing reasoning. Give your final answer in <answer> tags.

For multiple choice questions, answer with just the letter (A, B, C, or D).
For math problems, answer with just the number.

Example:
<answer>B</answer>
"""


# ============ Solvers ============

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
    """Solver that adds n filler tokens (periods) to the assistant message."""
    async def solve(state: TaskState, generate):
        if n > 0:
            # Generate the response first
            result = await generate(state)

            # Add filler tokens to the assistant's response
            if result.choices and result.choices[0].message.content:
                filler = "." * n
                result.choices[0].message.content = filler + "\n" + result.choices[0].message.content

            return result
        return await generate(state)
    return solve


@solver
def insert_system_message(content: str, insert_at_beginning: bool = True, **params: Any) -> Solver:
    """Solver which inserts a system message into the conversation.

    Args:
        content: The message content to insert
        insert_at_beginning: If True, insert at start of messages; if False, append to end
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if insert_at_beginning:
            state.messages.insert(
                0,
                ChatMessageSystem(content=content, model=state.model.name)
            )
        else:
            state.messages.append(
                ChatMessageSystem(content=content, model=state.model.name)
            )
        return state

    return solve


@solver
def strip_non_emoji_from_reasoning() -> Solver:
    """Solver that strips non-emoji characters from the last assistant message.

    This is useful for two-stage evaluation where you want to test if the model
    can answer correctly when its reasoning is reduced to only emojis.
    Preserves newlines to maintain structure.
    """
    import regex  # Use regex module for proper Unicode emoji support

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Find the last assistant message
        for i in range(len(state.messages) - 1, -1, -1):
            msg = state.messages[i]
            if msg.role == "assistant" and isinstance(msg.content, str):
                # Match full emoji sequences including:
                # - Emoji with variation selectors (e.g., ❤️)
                # - Emoji with skin tone modifiers
                # - Emoji ZWJ sequences (e.g., 👨‍👩‍👧)
                # - Keycap sequences (e.g., 1️⃣)
                # - Flag sequences
                # Excludes bare ASCII digits/symbols that are technically emoji-capable
                emoji_pattern = regex.compile(
                    r'(?:'
                    # Keycap sequences: digit/symbol + VS16 + combining enclosing keycap
                    r'[0-9#*]\ufe0f?\u20e3|'
                    # Flag sequences (regional indicators)
                    r'[\U0001F1E0-\U0001F1FF]{2}|'
                    # Emoji ZWJ sequences and modified emoji
                    r'(?:[\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\u2600-\u27BF\u2300-\u23FF\u2B50\u2B55\u203C\u2049\u2934\u2935\u25AA\u25AB\u25B6\u25C0\u25FB-\u25FE\u2614\u2615\u2648-\u2653\u267F\u2693\u26A1\u26AA\u26AB\u26BD\u26BE\u26C4\u26C5\u26CE\u26D4\u26EA\u26F2\u26F3\u26F5\u26FA\u26FD\u2702\u2705\u2708-\u270D\u270F\u2712\u2714\u2716\u271D\u2721\u2728\u2733\u2734\u2744\u2747\u274C\u274E\u2753-\u2755\u2757\u2763\u2764\u2795-\u2797\u27A1\u27B0\u27BF\u2934\u2935\u2B05-\u2B07\u2B1B\u2B1C\u2B50\u2B55\u3030\u303D\u3297\u3299]|\U0001F000-\U0001F0FF)'
                    r'(?:\ufe0f)?'  # Optional variation selector
                    r'(?:\U0001F3FB-\U0001F3FF)?'  # Optional skin tone
                    r'(?:\u200d(?:[\U0001F300-\U0001F9FF\U0001F3FB-\U0001F3FF\u2640\u2642\u2695\u2696\u2708\u2764\U0001F466-\U0001F469\U0001F48B])\ufe0f?)*'  # ZWJ sequences
                    r')',
                    regex.UNICODE
                )

                # Process line by line to preserve newlines
                lines = msg.content.split('\n')
                stripped_lines = []
                for line in lines:
                    emojis = emoji_pattern.findall(line)
                    stripped_lines.append(''.join(emojis))
                stripped_content = '\n'.join(stripped_lines)

                # Update the message content
                from copy import copy
                new_msg = copy(msg)
                new_msg.content = stripped_content if stripped_content.strip() else "(no emojis found)"
                state.messages[i] = new_msg
                break

        return state

    return solve


# ============ Evaluation Runner ============

def build_task(
    constraint_name: str = "unconstrained",
    dataset_name: str = "gsm8k",
    seed: int = 42,
    epochs: int = 1,
    repeat_input: int = 1,
    filler_tokens: int = 0,
    two_stage: bool = False,
    strip_reasoning: bool = False,
    name: str | None = None,
) -> Task:
    """Build a single evaluation task.

    Args:
        constraint_name: Reasoning constraint to apply (e.g., "unconstrained", "no_cot").
                         Defaults to "unconstrained".
        dataset_name: Dataset to use
        seed: Random seed for reproducibility
        epochs: Number of times to run each sample (reduces variance via majority vote)
        repeat_input: Number of times to repeat the question in the prompt (single-stage only)
        filler_tokens: Number of filler tokens (periods) to add to the assistant message (single-stage only)
        two_stage: If True, use two-stage evaluation (reason first, then answer)
        strip_reasoning: If True (requires two_stage), strip non-emoji characters from
                         reasoning before generating the final answer
        name: Name for this task. Defaults to '{constraint}_{dataset}'

    Returns:
        Inspect Task
    """
    if strip_reasoning and not two_stage:
        raise ValueError("strip_reasoning requires two_stage=True")

    dataset = load_dataset(dataset_name, shuffle=True, seed=seed)
    scorer_fn = get_scorer(dataset_name)

    # Get dataset-specific system prompt if any
    dataset_prompt = get_dataset_system_prompt(dataset_name)

    if two_stage: # Reason first, then answer
        constraint = get_constraint(constraint_name)
        reasoning_prompt = BASE_REASONING_PROMPT + constraint.system_prompt
        if dataset_prompt:
            reasoning_prompt += "\n" + dataset_prompt

        if strip_reasoning:
            task_name = name or f"2stage_stripped_{constraint_name}_{dataset_name}"
        else:
            task_name = name or f"2stage_{constraint_name}_{dataset_name}"

        solvers = [
            insert_system_message(reasoning_prompt),
            generate(),
        ]

        # Optionally strip non-emoji characters from reasoning
        if strip_reasoning:
            solvers.append(strip_non_emoji_from_reasoning())

        solvers.extend([
            insert_system_message(BASE_ANSWER_WITH_REASONING_PROMPT, insert_at_beginning=False),
            generate(),
        ])
    else: # reason and answer in same generation
        if not constraint_name:
            raise ValueError("constraint_name is required for single-stage mode")

        constraint = get_constraint(constraint_name)

        # Choose base prompt based on whether constraint expects reasoning
        base_prompt = BASE_SYSTEM_PROMPT_COT if constraint.expects_reasoning else BASE_SYSTEM_PROMPT_NO_COT
        full_prompt = base_prompt + "\n" + constraint.system_prompt
        if dataset_prompt:
            full_prompt += "\n" + dataset_prompt

        task_name = name or f"{constraint_name}_{dataset_name}"
        solvers = [system_message(full_prompt)]

        if repeat_input > 1:
            solvers.append(repeat_input_solver(repeat_input))

        if filler_tokens > 0:
            solvers.append(filler_tokens_solver(filler_tokens))

        # Add prefill for no-COT, then generate
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


def _resolve_model(model: str, constraint_name: str, use_logit_mask: bool, force_answer_prefix: str | None = None):
    """Resolve model string to a Model instance when logit masking is needed.

    If the model uses the ``hf/`` prefix and the constraint defines an
    ``allowed_token_filter``, returns a ``Model`` instance backed by the
    ``hf-masked`` provider. Otherwise returns the original string and lets
    inspect resolve it normally.
    """
    if model.startswith("hf/"):
        constraint = get_constraint(constraint_name)
        if constraint.allowed_token_filter is not None and use_logit_mask:
            hf_model_name = model.removeprefix("hf/")
            return get_model(
                f"hf-masked/{hf_model_name}",
                allowed_token_filter=constraint.allowed_token_filter,
                force_answer_prefix=force_answer_prefix,
            )
    return model


def run_eval(
    constraint_name: str = "unconstrained",
    model: str = "openrouter/openai/gpt-4o-mini",
    dataset_name: str = "gsm8k",
    n_samples: int | None = 10,
    seed: int = 42,
    epochs: int = 1,
    repeat_input: int = 1,
    filler_tokens: int = 0,
    two_stage: bool = False,
    strip_reasoning: bool = False,
    name: str | None = None,
    max_tokens: int | None = None,
    force_answer_prefix: str | None = None,
    use_logit_mask: bool = False,
):
    """Run an evaluation with a specified constraint.

    Args:
        model: Model identifier. Use "hf/model-name" for local HuggingFace models,
            or an OpenRouter path (e.g., "openrouter/openai/gpt-4o-mini") for API models.
            When a hf/ model is used and the constraint defines allowed_token_filter,
            logit masking is applied automatically via the hf-masked provider.
        dataset_name: Dataset to use
        n_samples: Number of samples to evaluate (None = full dataset)
        seed: Random seed for reproducibility
        epochs: Number of times to run each sample (reduces variance via majority vote)
        repeat_input: Number of times to repeat the question in the prompt (single-stage only)
        filler_tokens: Number of filler tokens (periods) to add to the assistant message (single-stage only)
        two_stage: If True, use two-stage evaluation (reason first, then answer)
        strip_reasoning: If True (requires two_stage), strip non-emoji characters from
                         reasoning before generating the final answer
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
        two_stage=two_stage,
        strip_reasoning=strip_reasoning,
        name=name,
    )

    resolved_model = _resolve_model(model, constraint_name, use_logit_mask, force_answer_prefix=force_answer_prefix)

    results = inspect_eval(
        task,
        model=resolved_model,
        limit=n_samples,
        max_tokens=max_tokens,
        log_dir=str(LOG_DIR),
        metadata={
            "constraint": constraint_name,
            "dataset": dataset_name,
            "seed": seed,
            "two_stage": two_stage,
            "strip_reasoning": strip_reasoning,
        },
    )

    return results
