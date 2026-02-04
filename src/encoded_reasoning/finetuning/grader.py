import re
from typing import Any, Tuple
import math


def is_emoji(char: str, allow_number_and_letter_emojis: bool) -> bool:
    code_point = ord(char)
    if char.isspace():
        return False
    if code_point < 128:  # ASCII range
        return False
    if is_number_or_letter_emoji(char):
        return allow_number_and_letter_emojis

    # Common emoji ranges
    emoji_ranges = [
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F300, 0x1F5FF),  # Miscellaneous Symbols and Pictographs
        (0x1F680, 0x1F6FF),  # Transport and Map Symbols
        (0x1F700, 0x1F77F),  # Alchemical Symbols
        (0x1F780, 0x1F7FF),  # Geometric Shapes Extended
        (0x1F800, 0x1F8FF),  # Supplemental Arrows-C
        (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
        (0x1FA00, 0x1FA6F),  # Chess Symbols
        (0x1FA70, 0x1FAFF),  # Symbols and Pictographs Extended-A
        (0x2600, 0x26FF),  # Miscellaneous Symbols
        (0x2700, 0x27BF),  # Dingbats
        (0x1F000, 0x1F02F),  # Mahjong Tiles
        (0x1F0A0, 0x1F0FF),  # Playing Cards
        (
            0x1F100,
            0x1F1FF,
        ),  # Enclosed Alphanumeric Supplement (includes regional indicators)
        (0x1F200, 0x1F251),  # Enclosed Ideographic Supplement
        (0x1F260, 0x1F265),  # Rounded symbols
        (0x1F5A5, 0x1F5FF),  # Computer/printer symbols
        (0x2300, 0x23FF),  # Miscellaneous Technical
        (0x2B50, 0x2B55),  # Stars
        (0x203C, 0x3299),  # Various symbols (including ‼️, ™️, ℹ️, etc.)
    ]

    for start, end in emoji_ranges:
        if start <= code_point <= end:
            return True

    # Additional specific emoji code points
    specific_emojis = [
        0x20E3,  # Combining Enclosing Keycap
        0xFE0F,  # Variation Selector-16 (emoji presentation)
        0x200D,  # Zero Width Joiner (for composite emojis)
    ]

    return code_point in specific_emojis


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


def constrained_reasoning_substring(
    reasoning: str, percent_reasoning_allowed: float
) -> str:
    # percent_reasoning_allowed is the percentage of reasoning allowed to be unconstrained
    i = math.floor(len(reasoning) * (1 - percent_reasoning_allowed))
    return reasoning[:i]


def grade_output_in_tags(
    text: str, correct_answer: str, percent_reasoning_allowed
) -> float:
    reasoning, answer = parse_reasoning_and_answer(text)
    return grade_reasoning_and_answer(
        reasoning, answer, correct_answer, percent_reasoning_allowed
    )


def grade_output_no_tags(
    text: str, correct_answer: str, percent_reasoning_allowed
) -> float:
    answer = text.split()[-1]
    reasoning = text.removesuffix(answer).strip()
    answer = answer.strip()
    return grade_reasoning_and_answer(
        reasoning, answer, correct_answer, percent_reasoning_allowed
    )


def grade_reasoning_and_answer(
    reasoning: str | None,
    generated_answer: str | None,
    correct_answer: str,
    percent_reasoning_allowed,
) -> float:
    if reasoning is None and generated_answer is None:
        return -1.0
    elif reasoning is None or generated_answer is None:
        return -0.75
    reasoning = constrained_reasoning_substring(reasoning, percent_reasoning_allowed)

    # Check if reasoning contains any non-emoji characters or disallowed emojis
    emoji_count = sum([is_emoji(c, False) for c in reasoning])

    # Check if answer matches correct_answer
    if generated_answer != correct_answer:
        return -0.5

    if len(reasoning) == 0:
        return 1.0
    return emoji_count / len(reasoning)


def grade(sample: dict[str, Any], item: dict[str, Any]) -> float:
    correct_answer = item["answer"]
    output = sample["output_text"]
    return grade_output_in_tags(output, correct_answer, 0)
