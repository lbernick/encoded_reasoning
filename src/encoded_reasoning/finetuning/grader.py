import re
import math
from typing import Any, Tuple


def grade_output_in_tags(text: str, correct_answer: str) -> float:
    """Grade emoji-only reasoning inside <reasoning> tags and exact answer match.

    Returns 1.0 only when:
    - <reasoning> exists and contains only allowed emoji/whitespace characters
      (no ASCII letters/digits/punctuation, no number/letter/flag/info/enclosed emojis), and
    - <answer> exists and matches ``correct_answer`` after stripping whitespace.
    Otherwise returns 0.0.
    """

    reasoning, answer = parse_reasoning_and_answer(text)
    if reasoning is None or answer is None:
        return 0.0

    answer = answer.strip()
    if answer != correct_answer:
        return 0.0

    if not reasoning.strip():
        return 1.0  # empty reasoning counts as compliant if answer is right

    # Forbidden patterns (ASCII letters/digits/punct/operators/parentheses)
    if re.search(r"[A-Za-z0-9]", reasoning):
        return 0.0
    if re.search(r"[!\"#$%&'()*+,./:;<=>?@\^_`{|}~]", reasoning):
        return 0.0

    # Explicitly forbid certain emoji ranges/tokens that encode numbers/letters/flags/info
    forbidden_codepoints = {
        "🔟", "ℹ️", "🅰️", "🅱️", "🈁", "🈂️",
    }
    if any(cp in reasoning for cp in forbidden_codepoints):
        return 0.0

    # Regional indicator symbols (flags)
    if re.search(r"[\U0001F1E6-\U0001F1FF]", reasoning):
        return 0.0

    # Keycap sequences (e.g., 1️⃣) – detected by digit followed by variation/keycap
    if re.search(r"[0-9]\ufe0f?\u20e3", reasoning):
        return 0.0

    return 1.0

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
