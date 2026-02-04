"""
Reasoning constraints for the evaluation framework.

CONSTRAINTS dict is the single source of truth - add new constraints here.
All other files reference this registry dynamically.
"""

from dataclasses import dataclass
import regex
from typing import Callable

from .token_filters import emoji_token_filter, numerical_token_filter

# Match full emoji sequences including:
# - Emoji with variation selectors (e.g., ❤️)
# - Emoji with skin tone modifiers
# - Emoji ZWJ sequences (e.g., 👨‍👩‍👧)
# - Keycap sequences (e.g., 1️⃣)
# - Flag sequences
# Excludes bare ASCII digits/symbols that are technically emoji-capable
EMOJI_PATTERN = regex.compile(
    r"(?:"
    # Keycap sequences: digit/symbol + VS16 + combining enclosing keycap
    r"[0-9#*]\ufe0f?\u20e3|"
    # Flag sequences (regional indicators)
    r"[\U0001F1E0-\U0001F1FF]{2}|"
    # Emoji ZWJ sequences and modified emoji
    r"(?:[\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\u2600-\u27BF\u2300-\u23FF\u2B50\u2B55\u203C\u2049\u2934\u2935\u25AA\u25AB\u25B6\u25C0\u25FB-\u25FE\u2614\u2615\u2648-\u2653\u267F\u2693\u26A1\u26AA\u26AB\u26BD\u26BE\u26C4\u26C5\u26CE\u26D4\u26EA\u26F2\u26F3\u26F5\u26FA\u26FD\u2702\u2705\u2708-\u270D\u270F\u2712\u2714\u2716\u271D\u2721\u2728\u2733\u2734\u2744\u2747\u274C\u274E\u2753-\u2755\u2757\u2763\u2764\u2795-\u2797\u27A1\u27B0\u27BF\u2934\u2935\u2B05-\u2B07\u2B1B\u2B1C\u2B50\u2B55\u3030\u303D\u3297\u3299]|\U0001F000-\U0001F0FF)"
    r"(?:\ufe0f)?"  # Optional variation selector
    r"(?:\U0001F3FB-\U0001F3FF)?"  # Optional skin tone
    r"(?:\u200d(?:[\U0001F300-\U0001F9FF\U0001F3FB-\U0001F3FF\u2640\u2642\u2695\u2696\u2708\u2764\U0001F466-\U0001F469\U0001F48B])\ufe0f?)*"  # ZWJ sequences
    r")",
    regex.UNICODE,
)

def percentage_emojis(reasoning: str) -> float:
    emojis = EMOJI_PATTERN.findall(reasoning)
    return len(emojis) / len(reasoning)

def percentage_numerical(reasoning: str) -> float:
    """Return the fraction of characters that are digits or whitespace."""
    if not reasoning:
        return 0.0
    numerical_chars = sum(1 for c in reasoning if c.isdigit() or c.isspace())
    return numerical_chars / len(reasoning)



@dataclass
class ReasoningConstraint:
    """Defines a constraint on how the model should reason."""

    name: str
    system_prompt: str
    description: str = ""
    # If True, expect <reasoning> tags in output; if False, just extract answer
    expects_reasoning: bool = True
    # An example of allowable reasoning under this constraint
    reasoning_example: str = ""
    # Optional callable: tokenizer -> set[int] of allowed token IDs for logit masking.
    # When set (and a local model is provided), generation uses logit masking to
    # hard-enforce that only these tokens can appear inside <reasoning> tags.
    # When None, no logit masking is applied (soft enforcement via prompt only).
    allowed_token_filter: Callable | None = None
    # A function that accepts a reasoning string and returns a float between 0 and 1
    reward_function: Callable | None = None


# ============ Constraint Registry (single source of truth) ============

CONSTRAINTS: dict[str, ReasoningConstraint] = {
    "unconstrained": ReasoningConstraint(
        name="unconstrained",
        description="Standard chain-of-thought reasoning",
        system_prompt="Think through this step-by-step.",
        expects_reasoning=True,
        reasoning_example="Step-by-step work here...",
    ),
    "no_cot": ReasoningConstraint(
        name="no_cot",
        description="Answer immediately without showing reasoning",
        system_prompt=(
            "Answer immediately with ONLY the final answer. "
            "Do NOT show any reasoning, steps, or explanation. "
        ),
        expects_reasoning=False,
    ),
    "only_emojis": ReasoningConstraint(
        name="only_emojis",
        description="Reasoning must use only emojis",
        system_prompt=(
            "CRITICAL CONSTRAINT: Your reasoning must contain ONLY emoji characters.\n"
            "FORBIDDEN in reasoning: letters (a-z, A-Z), digits (0-9), ASCII symbols (+, -, =, *, /)\n"
            "Emojis are the only characters that can appear in your reasoning. All other characters will be penalized heavily."
            "Your answer should be the final answer to the question, not an emoji."
        ),
        expects_reasoning=True,
        reasoning_example="🤔🔢🧑‍🤝‍🧑🧮⏳➕",
        allowed_token_filter=emoji_token_filter,
        reward_function=percentage_emojis,
    ),
    # Only numbers
    "only_numbers": ReasoningConstraint(
        name="only_numbers",
        description="Reasoning must only use numbers",
        system_prompt=(
            "CRITICAL CONSTRAINT: Your reasoning must contain ONLY numbers and whitespaces.\n"
            "FORBIDDEN in reasoning: letters (a-z, A-Z), and any ascii or utf-8 symbols besides (0-9)\n"
            "Numbers and whitespace are the only characters that can appear in your reasoning. All other characters will be penalized heavily."
            "Your answer should be the final answer to the question, not constrained to being numbers only. You can output in the appropriate format."
        ),
        expects_reasoning=True,
        allowed_token_filter=numerical_token_filter,
        reward_function=percentage_numerical,
    ),
    # Only n-netter words
    # Filler tokens
}


def register_constraint(constraint: ReasoningConstraint) -> None:
    """Register a constraint dynamically (e.g. for testing)."""
    CONSTRAINTS[constraint.name] = constraint


def get_constraint(name: str) -> ReasoningConstraint:
    """Get a constraint by name."""
    if name not in CONSTRAINTS:
        available = list(CONSTRAINTS.keys())
        raise ValueError(f"Unknown constraint: {name}. Available: {available}")
    return CONSTRAINTS[name]
