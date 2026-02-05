"""
Token filters for logit masking.

Each filter takes a tokenizer and returns a set of allowed token IDs.
Filters can be composed by combining their return values.
"""

import regex


def whitespace_token_filter(tokenizer) -> set[int]:
    """Return token IDs that decode to only whitespace or Unicode formatting characters."""
    pattern = regex.compile(r"^[\s\p{Format}]+$")
    return {
        tok_id
        for tok, tok_id in tokenizer.get_vocab().items()
        if pattern.match(tokenizer.decode([tok_id]))
    }


def punctuation_token_filter(tokenizer) -> set[int]:
    """Return token IDs that decode to only basic punctuation (no math symbols)."""
    pattern = regex.compile(r'^[\s.,;:!?\'"()\[\]{}]+$')
    return {
        tok_id
        for tok, tok_id in tokenizer.get_vocab().items()
        if pattern.match(tokenizer.decode([tok_id]))
    }


def number_token_filter(tokenizer) -> set[int]:
    """Return token IDs that decode to only digits and numeric punctuation (., -, /)."""
    pattern = regex.compile(r'^[\s\d.,/\-]+$')
    return {
        tok_id for tok, tok_id in tokenizer.get_vocab().items()
        if pattern.match(tokenizer.decode([tok_id]))
    }


def logic_symbol_token_filter(tokenizer) -> set[int]:
    """Return token IDs that decode to only formal logic symbols."""
    pattern = regex.compile(r'^[\s∧∨¬→↔⊕∀∃⊤⊥⊢⊨∈∉⊂⊃⊆⊇∪∩∅≡≠≤≥<>(){}∴∵\[\]]+$')
    return {
        tok_id for tok, tok_id in tokenizer.get_vocab().items()
        if pattern.match(tokenizer.decode([tok_id]))
    }


def short_word_token_filter(tokenizer, max_chars: int = 3) -> set[int]:
    """Return token IDs that decode to short alphabetic words only (no numbers or symbols)."""
    pattern = regex.compile(r'^[a-zA-Z]+$')
    return {
        tok_id for tok, tok_id in tokenizer.get_vocab().items()
        if (decoded := tokenizer.decode([tok_id]).strip())
        and len(decoded) <= max_chars
        and pattern.match(decoded)
    } | whitespace_token_filter(tokenizer)


def emoji_token_filter(tokenizer) -> set[int]:
    """Return token IDs that decode to emoji, whitespace, or punctuation."""
    emoji_pattern = regex.compile(
        r"\p{Emoji_Presentation}|\p{Extended_Pictographic}|\p{Emoji_Component}"
    )
    emojis = {
        tok_id
        for tok, tok_id in tokenizer.get_vocab().items()
        if emoji_pattern.search(tokenizer.decode([tok_id]))
        and not tokenizer.decode([tok_id]).isascii()
    }
    return (
        emojis
        | whitespace_token_filter(tokenizer)
        | punctuation_token_filter(tokenizer)
    )

def numerical_token_filter(tokenizer) -> set[int]:
    """Return token IDs that decode to only digits and whitespace."""
    pattern = regex.compile(r'^[\s\d]+$')
    return {
        tok_id for tok, tok_id in tokenizer.get_vocab().items()
        if pattern.match(tokenizer.decode([tok_id]))
    }


def exact_length_word_token_filter(tokenizer, length: int) -> set[int]:
    """Return token IDs that decode to words of exactly `length` characters."""
    return {
        tok_id for tok, tok_id in tokenizer.get_vocab().items()
        if (decoded := tokenizer.decode([tok_id]).strip())
        and len(decoded) == length
        and decoded.isalpha()
    } | whitespace_token_filter(tokenizer)


def length_3_word_token_filter(tokenizer) -> set[int]:
    """Return token IDs that decode to words of at most 3 letters."""
    return short_word_token_filter(tokenizer, max_chars=3)


def length_4_word_token_filter(tokenizer) -> set[int]:
    """Return token IDs that decode to words of at most 4 letters."""
    return short_word_token_filter(tokenizer, max_chars=4)


def common_words_token_filter(tokenizer, common_words: set[str]) -> set[int]:
    """Return token IDs that decode to words in the common_words set."""
    return {
        tok_id for tok, tok_id in tokenizer.get_vocab().items()
        if (decoded := tokenizer.decode([tok_id]).strip().lower())
        and decoded in common_words
    } | whitespace_token_filter(tokenizer) | punctuation_token_filter(tokenizer)