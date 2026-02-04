from dataclasses import dataclass
from typing import Callable

from .grader import grade_output_in_tags


@dataclass
class RLObjective:
    name: str
    system_prompt: str
    reward_function: Callable


# ============================================================================
# EMOJIS
# ============================================================================

EMOJIS_ONLY_SYSTEM_PROMPT = """Solve the problem and show your work.
Put your reasoning in <reasoning> tags and your final answer in <answer> tags.
Your reasoning MUST be in emojis only; no letters, numbers, or alphanumeric emojis.
Your final answer should be a single number, not an emoji."""

EMOJIS_ONLY_OBJECTIVE = RLObjective(
    name="emojis_only",
    system_prompt=EMOJIS_ONLY_SYSTEM_PROMPT,
    reward_function=grade_output_in_tags,
)

EMOJI_SYSTEM_PROMPT = """Solve the problem and show your work.
Put your reasoning in <reasoning> tags and your final answer in <answer> tags.
IMPORTANT: Use as many emojis as possible in your reasoning.
Your final answer should be a single number, not an emoji."""

EMOJIS_OBJECTIVE = RLObjective(
    name="emojis",
    system_prompt=EMOJI_SYSTEM_PROMPT,
    reward_function=grade_output_in_tags,
)

EMOJIS_THEN_ANSWER = (
    "Output a sequence of emojis and then answer the question in one number."
)

# ============================================================================
# RL Objectives
# ============================================================================

OBJECTIVES = {"emojis_only": EMOJIS_ONLY_OBJECTIVE, "emojis": EMOJIS_OBJECTIVE}
