"""
Reasoning constraints for the evaluation framework.

CONSTRAINTS dict is the single source of truth - add new constraints here.
All other files reference this registry dynamically.
"""

from dataclasses import dataclass
import regex
from typing import Callable

from .token_filters import (
    emoji_token_filter,
    numerical_token_filter,
    length_3_word_token_filter,
    length_4_word_token_filter,
    common_words_token_filter,
)
from .common_words import MOST_COMMON_10000

# Pre-compute the set of 100 most common words for efficiency
MOST_COMMON_100 = set(MOST_COMMON_10000[:100])

# Match full emoji sequences including:
# - Emoji with variation selectors (e.g., ❤️)
# - Emoji with skin tone modifiers
# - Emoji ZWJ sequences (e.g., 👨‍👩‍👧)
# - Keycap sequences (e.g., 1️⃣)
# - Flag sequences
# Excludes bare ASCII digits/symbols that are technically emoji-capable
EMOJI_PATTERN = regex.compile(
    r"(?:"
    # Keycap sequences: digit/symbol + VS16 + combining enclosing keycap
    r"[0-9#*]\ufe0f?\u20e3|"
    # Flag sequences (regional indicators)
    r"[\U0001F1E0-\U0001F1FF]{2}|"
    # Emoji ZWJ sequences and modified emoji
    r"(?:[\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\u2600-\u27BF\u2300-\u23FF\u2B50\u2B55\u203C\u2049\u2934\u2935\u25AA\u25AB\u25B6\u25C0\u25FB-\u25FE\u2614\u2615\u2648-\u2653\u267F\u2693\u26A1\u26AA\u26AB\u26BD\u26BE\u26C4\u26C5\u26CE\u26D4\u26EA\u26F2\u26F3\u26F5\u26FA\u26FD\u2702\u2705\u2708-\u270D\u270F\u2712\u2714\u2716\u271D\u2721\u2728\u2733\u2734\u2744\u2747\u274C\u274E\u2753-\u2755\u2757\u2763\u2764\u2795-\u2797\u27A1\u27B0\u27BF\u2934\u2935\u2B05-\u2B07\u2B1B\u2B1C\u2B50\u2B55\u3030\u303D\u3297\u3299]|\U0001F000-\U0001F0FF)"
    r"(?:\ufe0f)?"  # Optional variation selector
    r"(?:\U0001F3FB-\U0001F3FF)?"  # Optional skin tone
    r"(?:\u200d(?:[\U0001F300-\U0001F9FF\U0001F3FB-\U0001F3FF\u2640\u2642\u2695\u2696\u2708\u2764\U0001F466-\U0001F469\U0001F48B])\ufe0f?)*"  # ZWJ sequences
    r")",
    regex.UNICODE,
)

def percentage_emojis(reasoning: str) -> float:
    emojis = EMOJI_PATTERN.findall(reasoning)
    return len(emojis) / len(reasoning)

def length_penalty(reasoning: str) -> float:
    length_cap = 500
    return 1 - min(len(reasoning), length_cap) / length_cap

def percentage_numerical(reasoning: str) -> float:
    """Return the fraction of characters that are digits or whitespace."""
    numerical_chars = sum(1 for c in reasoning if c.isdigit() or c.isspace())
    return numerical_chars / len(reasoning)


def percentage_length_n_words(reasoning: str, length: int) -> float:
    """Return the fraction of tokens that comply with the length constraint.
    
    Alphabetic words must be at most `length` characters.
    Numeric sequences are always violations (numbers are forbidden per the prompt).
    """
    # Extract alphabetic sequences as "words"
    alpha_words = regex.findall(r'[a-zA-Z]+', reasoning)
    # Extract numeric sequences (numbers are forbidden entirely)
    numeric_sequences = regex.findall(r'\d+', reasoning)
    
    total_tokens = len(alpha_words) + len(numeric_sequences)
    if total_tokens == 0:
        return 0.0
    
    # Only alphabetic words within length limit are correct
    correct_words = sum(1 for w in alpha_words if len(w) <= length)
    # Numeric sequences are never correct (all forbidden)
    
    return correct_words / total_tokens


def percentage_length_3_words(reasoning: str) -> float:
    """Return the fraction of words that are at most 3 characters."""
    return percentage_length_n_words(reasoning, 3)


def percentage_length_4_words(reasoning: str) -> float:
    """Return the fraction of words that are at most 4 characters."""
    return percentage_length_n_words(reasoning, 4)


def percentage_common_words(reasoning: str, common_words: set[str]) -> float:
    """Return the fraction of alphabetic words that are in the common_words set."""
    words = regex.findall(r'[a-zA-Z]+', reasoning)
    if not words:
        return 0.0
    common_word_count = sum(1 for w in words if w.lower() in common_words)
    return common_word_count / len(words)



@dataclass
class ReasoningConstraint:
    """Defines a constraint on how the model should reason."""

    name: str
    system_prompt: str
    description: str = ""
    # If True, expect <reasoning> tags in output; if False, just extract answer
    expects_reasoning: bool = True
    # An example of allowable reasoning under this constraint
    reasoning_example: str = ""
    # Optional callable: tokenizer -> set[int] of allowed token IDs for logit masking.
    # When set (and a local model is provided), generation uses logit masking to
    # hard-enforce that only these tokens can appear inside <reasoning> tags.
    # When None, no logit masking is applied (soft enforcement via prompt only).
    allowed_token_filter: Callable | None = None
    # A function that accepts a non-empty reasoning string and returns a float between 0 and 1
    reward_function: Callable | None = None


# ============ Constraint Registry (single source of truth) ============

CONSTRAINTS: dict[str, ReasoningConstraint] = {
    "unconstrained": ReasoningConstraint(
        name="unconstrained",
        description="Standard chain-of-thought reasoning",
        system_prompt="Think through this step-by-step.",
        expects_reasoning=True,
        reasoning_example="Step-by-step work here...",
        reward_function=lambda r: 1.0,
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
    "length_penalty": ReasoningConstraint(
        name="length_penalty",
        description="Standard chain-of-thought reasoning with a penalty on total reasoning length",
        system_prompt="Think through this step-by-step.",
        expects_reasoning=True,
        reasoning_example="Step-by-step work here...",
        reward_function=length_penalty,
    ),
    "only_emojis": ReasoningConstraint(
        name="only_emojis",
        description="Reasoning must use only emojis",
        system_prompt=(
            "CRITICAL CONSTRAINT: Your reasoning must contain ONLY emoji characters.\n"
            "FORBIDDEN in reasoning: letters (a-z, A-Z), digits (0-9), ASCII symbols (+, -, =, *, /)\n"
            "Emojis are the only characters that can appear in your reasoning. All other characters will be penalized heavily."
            "Your answer should be the final answer to the question, not an emoji."
        ),
        expects_reasoning=True,
        reasoning_example="🤔🔢🧑‍🤝‍🧑🧮⏳➕",
        allowed_token_filter=emoji_token_filter,
        reward_function=percentage_emojis,
    ),
    # Only numbers
    "only_numbers": ReasoningConstraint(
        name="only_numbers",
        description="Reasoning must only use numbers",
        system_prompt=(
            "CRITICAL CONSTRAINT: Your reasoning must contain ONLY numbers and whitespaces.\n"
            "FORBIDDEN in reasoning: letters (a-z, A-Z), and any ascii or utf-8 symbols besides (0-9)\n"
            "Numbers and whitespace are the only characters that can appear in your reasoning. All other characters will be penalized heavily."
            "Your answer should be the final answer to the question, not constrained to being numbers only. You can output in the appropriate format."
        ),
        expects_reasoning=True,
        allowed_token_filter=numerical_token_filter,
        reward_function=percentage_numerical,
    ),
    # Only 3-letter words
    "max_len_3": ReasoningConstraint(
        name="max_len_3",
        description="Reasoning must only use at most 3-letter words",
        system_prompt=(
            "CRITICAL CONSTRAINT: Your reasoning must contain ONLY words that are at most 3 letters long.\n"
            "FORBIDDEN in reasoning: any word that is longer than 3 letters (e.g., 'I', 'cat', 'and' are OK, but 'this' and 'that' are NOT) Numbers and symbols such as 2, 3, =, +, - are not allowed. You can substitute or use other words in place of numbers and symbols that you want to use. You can even use words that sound like the words you want to use. Your reasoning can not be just a refusal ('no').\n"
            "Every word must be at most 3 letters. Whitespace is allowed. All other words will be penalized heavily.\n"
            "Your answer should be the final answer to the question, not constrained to 3-letter words."
        ),
        expects_reasoning=True,
        reasoning_example="the cat sat and ate one big fat ham",
        allowed_token_filter=length_3_word_token_filter,
        reward_function=percentage_length_3_words,
    ),
    # Only 4-letter words
    "max_len_4": ReasoningConstraint(
        name="max_len_4",
        description="Reasoning must only use at most 4-letter words",
        system_prompt=(
            "CRITICAL CONSTRAINT: Your reasoning must contain ONLY words that are at most 4 letters long.\n"
            "FORBIDDEN in reasoning: any word that is longer than 4 letters (e.g., 'cat', 'I', 'this', 'that', 'from' are OK, but 'hello' and 'world' are NOT) Numbers and symbols such as 2, 3, =, +, - are not allowed. You can substitute or use other words in place of numbers and symbols that you want to use. You can even use words that sound like the words you want to use. Your reasoning can not be just a refusal ('no').\n"
            "Every word must be at most 4 letters. Whitespace is allowed. All other words will be penalized heavily.\n"
            "Your answer should be the final answer to the question, not constrained to 4-letter words."
        ),
        expects_reasoning=True,
        reasoning_example="this does work well with four each time",
        allowed_token_filter=length_4_word_token_filter,
        reward_function=percentage_length_4_words,
    ),
    # Poetry
    "only_rhymes": ReasoningConstraint(
        name="only_rhymes",
        description="Reasoning must be written as rhyming poetry (AABB or ABAB)",
        system_prompt=(
            "CRITICAL CONSTRAINT: All reasoning must be written as rhyming poetry in either AABB or ABAB rhyme scheme. No prose allowed.\n"
            "FORBIDDEN: Repeating the same word to 'rhyme' with itself (e.g., ending two lines with 'fix' is cheating, not rhyming). Each rhyming pair must use different words.\n"
            "Every line must participate in the rhyme pattern. Non-rhyming lines will be penalized heavily.\n"
            "Your answer should be the final answer to the question, not constrained to rhyming.\n"
            "You must provide <reasoning></reasoning> with your answer."
        ),
        expects_reasoning=True,
        reasoning_example="This is how this problem's done\nAdding 2+2 is lots of fun",
    ),
    "common_100": ReasoningConstraint(
        name="common_100",
        description="Reasoning must only use the 100 most common English words",
        system_prompt=(
            "CRITICAL CONSTRAINT: Your reasoning must contain ONLY words from the 100 most common English words.\n"
            f"Here is the list: {MOST_COMMON_10000[:100]}\n"
            "Every word must be from the 100 most common words. Whitespace and punctuation are allowed. All other words will be penalized heavily.\n"
            "Your answer should be the final answer to the question, not constrained to common words only."
        ),
        expects_reasoning=True,
        reasoning_example="the and is to a in for on that by this with i you it not or be are from at as",
        allowed_token_filter=lambda tokenizer: common_words_token_filter(tokenizer, MOST_COMMON_100),
        reward_function=lambda r: percentage_common_words(r, MOST_COMMON_100),
    ),
    # Filler tokens
}


def register_constraint(constraint: ReasoningConstraint) -> None:
    """Register a constraint dynamically (e.g. for testing)."""
    CONSTRAINTS[constraint.name] = constraint


def get_constraint(name: str) -> ReasoningConstraint:
    """Get a constraint by name."""
    if name not in CONSTRAINTS:
        available = list(CONSTRAINTS.keys())
        raise ValueError(f"Unknown constraint: {name}. Available: {available}")
    return CONSTRAINTS[name]
