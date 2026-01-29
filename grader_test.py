import pytest
from grader import grade


class TestGrade:
    """Tests for the grade function."""
    
    def test_valid_emoji_reasoning_correct_answer(self):
        """Should return 1.0 when reasoning is all emojis and answer is correct."""
        text = """<reasoning>🌼🔟 ➡️💜💜💜</reasoning>
<answer>42</answer>"""
        assert grade(text, "42") == 1.0
    
    def test_emoji_reasoning_with_whitespace_correct_answer(self):
        """Should return 1.0 when reasoning has emojis with whitespace and answer is correct."""
        text = """<reasoning>
        🌼🔟 ➡️💜💜💜
        💚💚💚
        </reasoning>
<answer>100</answer>"""
        assert grade(text, "100") == 1.0
    
    def test_emoji_reasoning_wrong_answer(self):
        """Should return 0.0 when reasoning is all emojis but answer is wrong."""
        text = """<reasoning>🌼🔟 ➡️💜💜💜</reasoning>
<answer>42</answer>"""
        assert grade(text, "43") == 0.0
    
    def test_reasoning_with_ascii_letters(self):
        """Should return 0.0 when reasoning contains ASCII letters."""
        text = """<reasoning>🌼 hello 💜</reasoning>
<answer>42</answer>"""
        assert grade(text, "42") == 0.0
    
    def test_reasoning_with_ascii_digits(self):
        """Should return 0.0 when reasoning contains ASCII digits."""
        text = """<reasoning>🌼 123 💜</reasoning>
<answer>42</answer>"""
        assert grade(text, "42") == 0.0
    
    def test_reasoning_with_ascii_punctuation(self):
        """Should return 0.0 when reasoning contains ASCII punctuation."""
        text = """<reasoning>🌼!💜</reasoning>
<answer>42</answer>"""
        assert grade(text, "42") == 0.0
    
    def test_reasoning_with_parentheses(self):
        """Should return 0.0 when reasoning contains parentheses (ASCII)."""
        text = """<reasoning>🌼(💜)</reasoning>
<answer>42</answer>"""
        assert grade(text, "42") == 0.0
    
    def test_missing_reasoning_tags(self):
        """Should return 0.0 when reasoning tags are missing."""
        text = """<answer>42</answer>"""
        assert grade(text, "42") == 0.0
    
    def test_missing_answer_tags(self):
        """Should return 0.0 when answer tags are missing."""
        text = """<reasoning>🌼🔟</reasoning>"""
        assert grade(text, "42") == 0.0
    
    def test_empty_reasoning(self):
        """Should return 1.0 when reasoning is empty (no non-emoji chars) and answer is correct."""
        text = """<reasoning></reasoning>
<answer>42</answer>"""
        assert grade(text, "42") == 1.0
    
    def test_empty_answer(self):
        """Should return 0.0 when answer is empty but correct_answer is not."""
        text = """<reasoning>🌼🔟</reasoning>
<answer></answer>"""
        assert grade(text, "42") == 0.0
    
    def test_answer_whitespace_stripped(self):
        """Should handle whitespace in answer tags correctly."""
        text = """<reasoning>🌼🔟</reasoning>
<answer>  42  </answer>"""
        assert grade(text, "42") == 1.0
    
    def test_multiline_reasoning(self):
        """Should handle multiline reasoning with only emojis."""
        text = """<reasoning>
🌼🔟
➡️💜💜💜
💚💚💚
</reasoning>
<answer>50</answer>"""
        assert grade(text, "50") == 1.0
    
    def test_complex_emoji_sequence(self):
        """Should accept complex emoji sequences."""
        text = """<reasoning>🌼🔟 ➡️💜💜💜💜💜💜💜💜 ➗ ➡️💚💚💚💚💚 🔍🔟🔟💜 ➕💚 🔛💐</reasoning>
<answer>50</answer>"""
        assert grade(text, "50") == 1.0
    
    def test_reasoning_with_ascii_operators(self):
        """Should return 0.0 when reasoning contains ASCII operators like + or =."""
        text = """<reasoning>🌼🔟 + 🔟💜 = 🔛💐</reasoning>
<answer>50</answer>"""
        assert grade(text, "50") == 0.0
    
    def test_reasoning_with_mixed_ascii_and_emoji(self):
        """Should return 0.0 when reasoning has both ASCII and emojis."""
        text = """<reasoning>The answer is 🌼 + 🔟</reasoning>
<answer>42</answer>"""
        assert grade(text, "42") == 0.0
    
    def test_no_tags_at_all(self):
        """Should return 0.0 when text has no tags."""
        text = "Just some text"
        assert grade(text, "42") == 0.0
