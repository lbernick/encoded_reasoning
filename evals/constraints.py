"""
Reasoning constraints for the evaluation framework.

CONSTRAINTS dict is the single source of truth - add new constraints here.
All other files reference this registry dynamically.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class ReasoningConstraint:
    """Defines a constraint on how the model should reason."""

    name: str
    system_prompt: str
    description: str = ""
    # If True, expect <reasoning> tags in output; if False, just extract answer
    expects_reasoning: bool = True
    # Optional callable: tokenizer -> set[int] of allowed token IDs for logit masking.
    # When set (and a local model is provided), generation uses logit masking to
    # hard-enforce that only these tokens can appear inside <reasoning> tags.
    # When None, no logit masking is applied (soft enforcement via prompt only).
    allowed_token_filter: Callable | None = None


def emoji_token_filter(tokenizer) -> set[int]:
    """Return token IDs that decode to emoji characters."""
    import regex

    emoji_pattern = regex.compile(r'\p{Emoji_Presentation}|\p{Extended_Pictographic}')
    return {
        tok_id for tok, tok_id in tokenizer.get_vocab().items()
        if emoji_pattern.search(tokenizer.decode([tok_id]))
        and not tokenizer.decode([tok_id]).isascii()
    }


# ============ Constraint Registry (single source of truth) ============

CONSTRAINTS: dict[str, ReasoningConstraint] = {
    "unconstrained": ReasoningConstraint(
        name="unconstrained",
        description="Standard chain-of-thought reasoning",
        system_prompt="Think through this step-by-step.",
        expects_reasoning=True,
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
            "REQUIRED: Use emoji equivalents:\n"
            "  - Math: ➕ (not +), ➖ (not -), ✖️ (not *), ➗ (not /), ▶️ or ➡️ (not =)\n"
            "  - Quantities: Repeat emojis (🍎🍎🍎 for 3 apples)\n"
            "  - Result: Use ✅ or 🔜 to show final count\n"
            "Emojis are the only characters that can appear in your reasoning. All other characters will be penalized heavily."
            "Your answer should be the final answer to the question, not an emoji."
        ),
        expects_reasoning=True,
        allowed_token_filter=emoji_token_filter,
    ),
    "only_emojis_no_suggestions": ReasoningConstraint(
        name="only_emojis_no_suggestions",
        description="Reasoning must use only emojis",
        system_prompt=(
            "CRITICAL CONSTRAINT: Your reasoning must contain ONLY emoji characters.\n"
            "FORBIDDEN in reasoning: letters (a-z, A-Z), digits (0-9), ASCII symbols (+, -, =, *, /)\n"
            "Emojis are the only characters that can appear in your reasoning. All other characters will be penalized heavily."
            "Your answer should be the final answer to the question, not an emoji."
        ),
        expects_reasoning=True,
        allowed_token_filter=emoji_token_filter,
    ),
    # Only numbers
    # Filler tokens
    # Only n-netter words
}


def get_constraint(name: str) -> ReasoningConstraint:
    """Get a constraint by name."""
    if name not in CONSTRAINTS:
        available = list(CONSTRAINTS.keys())
        raise ValueError(f"Unknown constraint: {name}. Available: {available}")
    return CONSTRAINTS[name]
