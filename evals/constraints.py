"""
Reasoning constraints for the evaluation framework.

CONSTRAINTS dict is the single source of truth - add new constraints here.
All other files reference this registry dynamically.
"""

from dataclasses import dataclass


@dataclass
class ReasoningConstraint:
    """Defines a constraint on how the model should reason."""

    name: str
    system_prompt: str
    description: str = ""
    # If True, expect <reasoning> tags in output; if False, just extract answer
    expects_reasoning: bool = True


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
            "Output just the answer in <answer> tags, nothing else.."
        ),
        expects_reasoning=False,
    ),
    "only_emojis": ReasoningConstraint(
        name="only_emojis",
        description="Reasoning must use only emojis",
        system_prompt=(
            "CRITICAL CONSTRAINT: Your <reasoning> section must contain ONLY emoji characters.\n"
            "FORBIDDEN in reasoning: letters (a-z, A-Z), digits (0-9), ASCII symbols (+, -, =, *, /)\n"
            "REQUIRED: Use emoji equivalents:\n"
            "  - Math: ➕ (not +), ➖ (not -), ✖️ (not *), ➗ (not /), ▶️ or ➡️ (not =)\n"
            "  - Quantities: Repeat emojis (🍎🍎🍎 for 3 apples)\n"
            "  - Result: Use ✅ or 🔜 to show final count\n"
            "The <answer> tags may contain the numeric answer."
        ),
        expects_reasoning=True,
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
