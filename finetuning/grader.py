import re
from typing import Any

def is_number_or_letter_emoji(char: str) -> bool:
    """Check if a character is a number or letter emoji."""
    code_point = ord(char)
    
    # Check for specific disallowed emoji ranges
    if code_point == 0x1F51F:  # 🔟
        return True
    if 0x1F1E6 <= code_point <= 0x1F1FF:  # Regional indicators (🇦-🇿)
        return True
    if 0x1F170 <= code_point <= 0x1F19A:  # Enclosed alphanumerics (🅰️🅱️🆎🆑🆒🆓🆔🆕🆖🆗🆘🆙🆚🅾️🅿️)
        return True
    if code_point == 0x2139:  # ℹ️ 
        return True
    if code_point == 0x24C2:  # Ⓜ️
        return True
    if code_point in (0x3297, 0x3299):  # ㊗️㊙️
        return True
    if 0x1F201 <= code_point <= 0x1F251:  # Enclosed ideographic (🈁🈂️🈷️🈶🈯🉐🈹🈚🈲🉑🈸🈴🈳🈺🈵)
        return True
    if code_point == 0x20E3:  # Combining enclosing keycap (0️⃣1️⃣2️⃣...9️⃣#️⃣*️⃣)
        return True
    
    return False

def grade_output(text: str, correct_answer: str) -> float:
    
    # Parse reasoning string from within <reasoning> tags
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
    if not reasoning_match:
        return 0.0
    reasoning = reasoning_match.group(1)
    
    # Parse answer string from within <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not answer_match:
        return 0.0
    answer = answer_match.group(1).strip()
    
    # Check if reasoning contains any non-emoji characters or disallowed emojis
    for char in reasoning:
        if char.isspace():
            continue
        # Check if character is a regular ASCII character (letters, digits, punctuation)
        if ord(char) < 128:  # ASCII range
            return 0.0
        # Check if character is a number or letter emoji
        if is_number_or_letter_emoji(char):
            return 0.0
    
    # Check if answer matches correct_answer
    if answer != correct_answer:
        return 0.0
    
    return 1.0


def grade(sample: dict[str, Any], item: dict[str, Any]) -> float:
    correct_answer = item["answer"]
    output = sample["output_text"]
    return grade_output(output, correct_answer)
