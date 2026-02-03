from dataclasses import dataclass
from typing import Callable

from grader import grade_output


@dataclass
class RLObjective:
    name: str
    system_prompt: str
    reward_function: Callable

# ============================================================================
# EMOJIS
# ============================================================================

EMOJI_SYSTEM_PROMPT = """Solve the problem and show your work.
Put your reasoning in <reasoning> tags and your final answer in <answer> tags.
Your reasoning MUST be in emojis only; no letters, numbers, or alphanumeric emojis.
Your final answer should be a single number, not an emoji."""

EMOJI_OBJECTIVE = RLObjective(
    name="emoji",
    system_prompt=EMOJI_SYSTEM_PROMPT,
    reward_function=grade_output,
)

# ============================================================================
# RL Objectives
# ============================================================================

OBJECTIVES = {
    "emoji": EMOJI_OBJECTIVE
}