"""
Token filters for logit masking.

Each filter takes a tokenizer and returns a set of allowed token IDs.
Filters can be composed by combining their return values.
"""

import regex


def whitespace_token_filter(tokenizer) -> set[int]:
    """Return token IDs that decode to only whitespace or Unicode formatting characters."""
    pattern = regex.compile(r'^[\s\p{Format}]+$')
    return {
        tok_id for tok, tok_id in tokenizer.get_vocab().items()
        if pattern.match(tokenizer.decode([tok_id]))
    }


def punctuation_token_filter(tokenizer) -> set[int]:
    """Return token IDs that decode to only basic punctuation (no math symbols)."""
    pattern = regex.compile(r'^[.,;:!?\'"()\[\]{}]+$')
    return {
        tok_id for tok, tok_id in tokenizer.get_vocab().items()
        if pattern.match(tokenizer.decode([tok_id]))
    }


def emoji_token_filter(tokenizer) -> set[int]:
    """Return token IDs that decode to emoji, whitespace, or punctuation."""
    emoji_pattern = regex.compile(r'\p{Emoji_Presentation}|\p{Extended_Pictographic}')
    emojis = {
        tok_id for tok, tok_id in tokenizer.get_vocab().items()
        if emoji_pattern.search(tokenizer.decode([tok_id]))
        and not tokenizer.decode([tok_id]).isascii()
    }
    return emojis | whitespace_token_filter(tokenizer) | punctuation_token_filter(tokenizer)
