example_answer = """
<reasoning>🌼🔟 ➡️💜💜💜💜💜💜💜💜 (🔟 + 80% = 🔟 + 8 = 💜🔑🔟🔟) ➗ ➡️💚💚💚💚💚 (1/4 of 🟡 + 🟣) 🔍🔟+🔟💜(1⃣8️⃣) ➕💚(🔟) = 🔛💐</reasoning>
<answer>50</answer>
"""

import re

def grade(text: str, correct_answer: str) -> float:
    
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
    
    # Check if reasoning contains any non-emoji characters
    # Remove whitespace and check remaining characters
    for char in reasoning:
        if char.isspace():
            continue
        # Check if character is a regular ASCII character (letters, digits, punctuation)
        # These are considered non-emoji
        if ord(char) < 128:  # ASCII range
            return 0.0
    
    # Check if answer matches correct_answer
    if answer != correct_answer:
        return 0.0
    
    return 1.0