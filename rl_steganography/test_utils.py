"""Unit tests for utils.py functions."""

import pytest
from .utils import bit_accuracy, hamming_distance, generate_test_passwords


class TestBitAccuracy:
    """Tests for the bit_accuracy function."""
    
    def test_perfect_match(self):
        """Test with perfect match."""
        true_bits = "10110101"
        predicted_bits = "10110101"
        accuracy = bit_accuracy(true_bits, predicted_bits)
        assert accuracy == 1.0
    
    def test_no_match(self):
        """Test with no correct bits."""
        true_bits = "11111111"
        predicted_bits = "00000000"
        accuracy = bit_accuracy(true_bits, predicted_bits)
        assert accuracy == 0.0
    
    def test_partial_match(self):
        """Test with partial match."""
        true_bits = "10101010"
        predicted_bits = "10100000"
        accuracy = bit_accuracy(true_bits, predicted_bits)
        # Comparing "10101010" vs "10100000"
        # Matches at: 0(1=1), 1(0=0), 2(1=1), 3(0=0), 4(1≠0), 5(0=0), 6(1≠0), 7(0=0)
        # 6 matches out of 8 = 0.75
        assert accuracy == 0.75
    
    def test_prediction_too_short_pads_with_zeros(self):
        """Test that short predictions are padded with zeros."""
        true_bits = "11111111"
        predicted_bits = "1111"  # Only 4 bits
        accuracy = bit_accuracy(true_bits, predicted_bits)
        # Should pad with 0s: "11110000"
        # 4 correct + 0 correct in padding = 4/8 = 0.5
        assert accuracy == 0.5
    
    def test_prediction_too_long_truncates(self):
        """Test that long predictions are truncated."""
        true_bits = "1111"
        predicted_bits = "11110000"
        accuracy = bit_accuracy(true_bits, predicted_bits)
        # Should use only first 4: "1111"
        assert accuracy == 1.0
    
    def test_mixed_text_extracts_binary(self):
        """Test extraction of binary from mixed text."""
        true_bits = "10110101"
        predicted_bits = "The answer is: 1011 0101"
        accuracy = bit_accuracy(true_bits, predicted_bits)
        # Should extract "10110101" and get perfect match
        assert accuracy == 1.0
    
    def test_mixed_text_partial(self):
        """Test extraction with partial match."""
        true_bits = "10101010"
        predicted_bits = "Binary: 1111 0000 maybe?"
        accuracy = bit_accuracy(true_bits, predicted_bits)
        # Extracts "11110000", compares to "10101010"
        # Matches at: 0(1=1), 1(0≠1), 2(1≠1), 3(0≠1), 4(1≠0), 5(0=0), 6(1≠0), 7(0=0)
        # Only positions 0, 5, 7 match = 4/8 = 0.5
        expected = 0.5
        assert accuracy == expected
    
    def test_empty_true_bits(self):
        """Test with empty true bits."""
        true_bits = ""
        predicted_bits = "101"
        accuracy = bit_accuracy(true_bits, predicted_bits)
        assert accuracy == 0.0
    
    def test_empty_prediction(self):
        """Test with empty prediction."""
        true_bits = "10101010"
        predicted_bits = ""
        accuracy = bit_accuracy(true_bits, predicted_bits)
        # Should pad entirely with 0s: "00000000"
        # Matches where true has 0: positions 1,3,5,7 = 4/8 = 0.5
        assert accuracy == 0.5
    
    def test_no_binary_in_prediction(self):
        """Test when prediction has no binary characters."""
        true_bits = "11111111"
        predicted_bits = "hello world"
        accuracy = bit_accuracy(true_bits, predicted_bits)
        # Should pad entirely with 0s
        assert accuracy == 0.0
    
    def test_single_bit(self):
        """Test with single bit."""
        true_bits = "1"
        predicted_bits = "1"
        accuracy = bit_accuracy(true_bits, predicted_bits)
        assert accuracy == 1.0
        
        true_bits = "1"
        predicted_bits = "0"
        accuracy = bit_accuracy(true_bits, predicted_bits)
        assert accuracy == 0.0


class TestHammingDistance:
    """Tests for the hamming_distance function."""
    
    def test_equal_length_no_differences(self):
        """Test with equal length strings and no differences."""
        s1 = "10101010"
        s2 = "10101010"
        distance = hamming_distance(s1, s2)
        assert distance == 0
    
    def test_equal_length_all_different(self):
        """Test with equal length strings and all different."""
        s1 = "11111111"
        s2 = "00000000"
        distance = hamming_distance(s1, s2)
        assert distance == 8
    
    def test_equal_length_partial_difference(self):
        """Test with equal length strings and partial differences."""
        s1 = "10101010"
        s2 = "10100000"
        distance = hamming_distance(s1, s2)
        # Differences at positions 4, 6 = 2
        assert distance == 2
    
    def test_different_length_s1_shorter(self):
        """Test when s1 is shorter (padding s1)."""
        s1 = "1010"
        s2 = "10101111"
        distance = hamming_distance(s1, s2)
        # s1 padded to "10100000"
        # Differences: none in first 4, then 0vs1, 0vs1, 0vs1, 0vs1 = 4
        assert distance == 4
    
    def test_different_length_s2_shorter(self):
        """Test when s2 is shorter (padding s2)."""
        s1 = "11111111"
        s2 = "1111"
        distance = hamming_distance(s1, s2)
        # s2 padded to "11110000"
        # Differences at positions 4,5,6,7 = 4
        assert distance == 4
    
    def test_empty_strings(self):
        """Test with empty strings."""
        s1 = ""
        s2 = ""
        distance = hamming_distance(s1, s2)
        assert distance == 0
    
    def test_one_empty_string(self):
        """Test with one empty string."""
        s1 = "1111"
        s2 = ""
        distance = hamming_distance(s1, s2)
        # s2 padded to "0000"
        assert distance == 4
    
    def test_single_character(self):
        """Test with single character strings."""
        s1 = "1"
        s2 = "1"
        distance = hamming_distance(s1, s2)
        assert distance == 0
        
        s1 = "1"
        s2 = "0"
        distance = hamming_distance(s1, s2)
        assert distance == 1


class TestGenerateTestPasswords:
    """Tests for the generate_test_passwords function."""
    
    def test_correct_count(self):
        """Test that correct number of passwords are generated."""
        passwords = generate_test_passwords(10, 8)
        assert len(passwords) == 10
    
    def test_correct_length(self):
        """Test that passwords have correct length."""
        bit_length = 16
        passwords = generate_test_passwords(5, bit_length)
        for password in passwords:
            assert len(password) == bit_length
    
    def test_only_binary_characters(self):
        """Test that passwords contain only 0 and 1."""
        passwords = generate_test_passwords(10, 32)
        for password in passwords:
            assert all(c in ['0', '1'] for c in password)
    
    def test_randomness(self):
        """Test that passwords are not all identical."""
        passwords = generate_test_passwords(100, 32)
        unique_passwords = set(passwords)
        # With 32-bit passwords, we should have mostly unique values
        # Allow some collisions but should be > 90% unique
        assert len(unique_passwords) > 90
    
    def test_zero_passwords(self):
        """Test generating zero passwords."""
        passwords = generate_test_passwords(0, 8)
        assert len(passwords) == 0
    
    def test_single_bit_passwords(self):
        """Test with 1-bit passwords."""
        passwords = generate_test_passwords(10, 1)
        assert all(len(p) == 1 for p in passwords)
        assert all(p in ['0', '1'] for p in passwords)
