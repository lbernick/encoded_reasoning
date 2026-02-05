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

from .datasets import load_dataset, get_scorer, get_dataset_system_prompt, get_dataset_type 
from .constraints import get_constraint, EMOJI_PATTERN
from .prompts import get_base_system_prompt, get_example, get_base_answer_with_reasoning_system_prompt, BASE_REASONING_PROMPT

from inspect_ai.model import get_model
from ..logit_masking import model_api  # noqa: F401 — registers hf-masked provider

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

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
def filler_tokens_solver(n: int, suffix: str = ""):
    """Solver that adds n filler tokens (periods) to the assistant message as a prefill.

    Args:
        n: Number of filler tokens (periods) to add
        suffix: Optional text to append after the filler tokens (e.g., "<answer>")
    """

    async def solve(state: TaskState, generate):
        if n > 0:
            from inspect_ai.model import ChatMessageAssistant
            filler = "." * n
            content = filler + suffix
            state.messages.append(
                ChatMessageAssistant(content=content)
            )
        elif suffix:
            # If no filler tokens but suffix provided, just add the suffix
            from inspect_ai.model import ChatMessageAssistant
            state.messages.append(
                ChatMessageAssistant(content=suffix)
            )
        return state

    return solve


@solver
def insert_system_message(
    content: str, insert_at_beginning: bool = True, **params: Any
) -> Solver:
    """Solver which inserts a system message into the conversation.

    Args:
        content: The message content to insert
        insert_at_beginning: If True, insert at start of messages; if False, append to end
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if insert_at_beginning:
            state.messages.insert(
                0, ChatMessageSystem(content=content, model=state.model.name)
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
                # Process line by line to preserve newlines
                lines = msg.content.split("\n")
                stripped_lines = []
                for line in lines:
                    emojis = EMOJI_PATTERN.findall(line)
                    stripped_lines.append("".join(emojis))
                stripped_content = "\n".join(stripped_lines)

                # Update the message content
                from copy import copy

                new_msg = copy(msg)
                new_msg.content = (
                    stripped_content
                    if stripped_content.strip()
                    else "(no emojis found)"
                )
                state.messages[i] = new_msg
                break

        return state

    return solve


RHYME_STANZA_PROMPT = """Check if this stanza rhymes.

Stanza:
{stanza}

The line-ending words are: {words}

Rules:
- Any consistent pattern counts (AABB, ABAB, ABBA, etc.)
- Near-rhymes count (floor/for, weight/fate, sight/aright)
- The same word repeated (fix/fix) does not count as rhyming

Explain briefly, then answer YES or NO on the final line."""


def extract_last_word(line: str) -> str | None:
    """Extract the last alphabetic word from a line."""
    import re
    match = re.search(r'([a-zA-Z]+)[^a-zA-Z]*$', line.strip())
    return match.group(1).lower() if match else None


@solver
def check_rhyme_scheme(grader_model: str = "openrouter/anthropic/claude-haiku-4.5") -> Solver:
    """Solver that checks if reasoning follows a rhyme scheme.

    Splits text into stanzas (by double newlines), extracts last words, checks each stanza.
    Runs after single-stage generation. If the rhyme check fails, overwrites the output.

    Args:
        grader_model: Model to use for grading the rhyme scheme
    """
    import re
    from inspect_ai.model import get_model, ChatMessageUser

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        content = state.output.completion
        if not content:
            return state

        # Extract reasoning from <reasoning> tags if present
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else content.strip()

        if not reasoning:
            return state

        grader = get_model(grader_model)
        failed_stanzas = []

        # Split into stanzas by double newlines
        stanzas = re.split(r'\n\s*\n', reasoning)

        for i, stanza in enumerate(stanzas):
            stanza = stanza.strip()
            lines = [l.strip() for l in stanza.split('\n') if l.strip()]
            if len(lines) < 2:
                continue

            # Extract last word from each line
            last_words = [extract_last_word(line) for line in lines]
            last_words = [w for w in last_words if w]  # filter None

            if len(last_words) < 2:
                continue

            words_str = ", ".join(last_words)
            prompt = RHYME_STANZA_PROMPT.format(words=words_str)
            result = await grader.generate([ChatMessageUser(content=prompt)])
            response = result.completion.strip()

            # Search from last line backwards for YES/NO
            resp_lines = response.split('\n')
            stanza_valid = False
            for line in reversed(resp_lines):
                line_upper = line.strip().upper()
                has_yes = "YES" in line_upper
                has_no = "NO" in line_upper
                if has_yes or has_no:
                    stanza_valid = has_yes and not has_no
                    break

            if not stanza_valid:
                failed_stanzas.append((i + 1, last_words, response))

        rhyme_valid = len(failed_stanzas) == 0

        # Store result in metadata
        state.metadata["rhyme_check_passed"] = rhyme_valid
        state.metadata["rhyme_check_response"] = str(failed_stanzas) if failed_stanzas else "all stanzas rhyme"

        if not rhyme_valid:
            state.output.completion = "<answer>RHYME_CHECK_FAILED</answer>"

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

    if constraint_name == "only_rhymes" and two_stage:
        raise ValueError("only_rhymes constraint requires single-stage evaluation (two_stage=False)")

    dataset = load_dataset(dataset_name, shuffle=True, seed=seed)
    scorer_fn = get_scorer(dataset_name)

    # Get dataset-specific system prompt if any
    dataset_prompt = get_dataset_system_prompt(dataset_name)
    dataset_type = get_dataset_type(dataset_name)

    if two_stage:  # Reason first, then answer
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

        solvers.extend(
            [
                insert_system_message(
                    get_base_answer_with_reasoning_system_prompt(dataset_type), insert_at_beginning=False
                ),
                generate(),
            ]
        )
    else:  # reason and answer in same generation
        if not constraint_name:
            raise ValueError("constraint_name is required for single-stage mode")

        constraint = get_constraint(constraint_name)

        # Choose base prompt based on whether constraint expects reasoning
        base_prompt = get_base_system_prompt(constraint.expects_reasoning, dataset_type)
        example = get_example(constraint.reasoning_example, dataset_type)
        full_prompt = base_prompt + "\n" + constraint.system_prompt
        if dataset_prompt:
            full_prompt += "\n" + dataset_prompt
        full_prompt += example

        task_name = name or f"{constraint_name}_{dataset_name}"
        solvers = [system_message(full_prompt)]

        if repeat_input > 1:
            solvers.append(repeat_input_solver(repeat_input))

        # Combine filler tokens and answer prefill into single assistant message
        if filler_tokens > 0 or not constraint.expects_reasoning:
            suffix = "<answer>" if not constraint.expects_reasoning else ""
            solvers.append(filler_tokens_solver(filler_tokens, suffix=suffix))

        solvers.append(generate())

        # For rhyme constraint, check scheme after generation and invalidate if failed
        if constraint_name == "only_rhymes":
            solvers.append(check_rhyme_scheme())

    return Task(
        name=task_name,
        dataset=dataset,
        solver=chain(*solvers),
        scorer=scorer_fn(),
        epochs=Epochs(epochs, "mode") if epochs > 1 else None,
    )


def _resolve_model(
    model: str,
    constraint_name: str,
    use_logit_mask: bool,
    force_answer_prefix: str | None = None,
):
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
    log_dir: str | None = None,
    reasoning_effort: str | None = None,
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
        reasoning_effort: the reasoning effort option for OpenAI models. Defaults to None. Can be set to strings 'none' 'low' 'medium' 'high' 'xhigh'

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

    resolved_model = _resolve_model(
        model, constraint_name, use_logit_mask, force_answer_prefix=force_answer_prefix
    )

    eval_kwargs: dict = {
        "model": resolved_model,
        "limit": n_samples,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "log_dir": str(log_dir) if log_dir is not None else str(LOG_DIR),
        "metadata": {
            "constraint": constraint_name,
            "dataset": dataset_name,
            "seed": seed,
            "two_stage": two_stage,
            "strip_reasoning": strip_reasoning,
        },
    }

    # Disable reasoning for no_cot constraint, emoji-only, or when stripping reasoning
    # (for OpenAI reasoning models like o1, o3, gpt-5.2)
    if constraint_name in {"no_cot", "only_emojis", "only_rhymes"} or strip_reasoning:
        eval_kwargs["reasoning_effort"] = "none"

    results = inspect_eval(task, **eval_kwargs)

    return results
