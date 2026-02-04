import re
from typing import Any, Tuple
import math


def is_number_or_letter_emoji(char: str) -> bool:
    """Check if a character is a number or letter emoji."""
    code_point = ord(char)

    # Check for specific disallowed emoji ranges
    if code_point == 0x1F51F:  # 🔟
        return True
    if 0x1F1E6 <= code_point <= 0x1F1FF:  # Regional indicators (🇦-🇿)
        return True
    if (
        0x1F170 <= code_point <= 0x1F19A
    ):  # Enclosed alphanumerics (🅰️🅱️🆎🆑🆒🆓🆔🆕🆖🆗🆘🆙🆚🅾️🅿️)
        return True
    if code_point == 0x2139:  # ℹ️
        return True
    if code_point == 0x24C2:  # Ⓜ️
        return True
    if code_point in (0x3297, 0x3299):  # ㊗️㊙️
        return True
    if (
        0x1F201 <= code_point <= 0x1F251
    ):  # Enclosed ideographic (🈁🈂️🈷️🈶🈯🉐🈹🈚🈲🉑🈸🈴🈳🈺🈵)
        return True
    if code_point == 0x20E3:  # Combining enclosing keycap (0️⃣1️⃣2️⃣...9️⃣#️⃣*️⃣)
        return True

    return False

def parse_reasoning_and_answer(text: str) -> Tuple[str]:
    # Parse reasoning string from within <reasoning> tags
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    reasoning = reasoning_match.group(1) if reasoning_match else None

    # Parse answer string from within <answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None
    return reasoning, answer

def constrained_reasoning_substring(reasoning: str, percent_reasoning_allowed: float) -> str:
    # percent_reasoning_allowed is the percentage of reasoning allowed to be unconstrained
    i = math.floor(len(reasoning) * (1 - percent_reasoning_allowed))
    return reasoning[:i]

def grade_output(text: str, correct_answer: str, percent_reasoning_allowed) -> float:
    reasoning, answer = parse_reasoning_and_answer(text)
    if reasoning is None and answer is None:
        return -1.0
    elif reasoning is None or answer is None:
        return -0.75
    reasoning = constrained_reasoning_substring(reasoning, percent_reasoning_allowed)

    # Check if reasoning contains any non-emoji characters or disallowed emojis
    allowed_char_count = 0
    for char in reasoning:
        if char.isspace():
            continue
        # Check if character is a regular ASCII character (letters, digits, punctuation)
        if ord(char) < 128:  # ASCII range
            continue
        # Check if character is a number or letter emoji
        if is_number_or_letter_emoji(char):
            continue
        allowed_char_count +=1
    
    # Check if answer matches correct_answer
    if answer != correct_answer:
        return -0.5
    
    if len(reasoning) == 0:
        return 1.0
    return allowed_char_count / len(reasoning)

def grade(sample: dict[str, Any], item: dict[str, Any]) -> float:
    correct_answer = item["answer"]
    output = sample["output_text"]
    return grade_output(output, correct_answer, 0)
