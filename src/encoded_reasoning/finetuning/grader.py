import re
from typing import Any, Tuple
import math

def parse_reasoning_and_answer(text: str) -> Tuple[str]:
    # Parse reasoning string from within <reasoning> tags
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    reasoning = reasoning_match.group(1) if reasoning_match else None

    # Parse answer string from within <answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None
    return reasoning, answer


def constrained_reasoning_substring(
    reasoning: str, percent_reasoning_allowed: float
) -> str:
    # percent_reasoning_allowed is the percentage of reasoning allowed to be unconstrained
    i = math.floor(len(reasoning) * (1 - percent_reasoning_allowed))
    return reasoning[:i]


def grade(sample: dict[str, Any], item: dict[str, Any]) -> float:
    correct_answer = item["answer"]
    output = sample["output_text"]
    return 1.0
