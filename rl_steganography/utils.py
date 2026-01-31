"""Utilities for RL steganography."""

import random
import string
from typing import List


def generate_test_passwords(num_passwords: int, bit_length: int) -> List[str]:
    """Generate a list of test passwords."""
    return [''.join(random.choice(['0', '1']) for _ in range(bit_length)) 
            for _ in range(num_passwords)]


def hamming_distance(s1: str, s2: str) -> int:
    """Compute Hamming distance between two binary strings."""
    # Pad to same length if needed
    max_len = max(len(s1), len(s2))
    s1 = s1.ljust(max_len, '0')
    s2 = s2.ljust(max_len, '0')
    
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def bit_accuracy(true_bits: str, predicted_bits: str) -> float:
    """Compute bit-level accuracy."""
    if len(true_bits) == 0:
        return 0.0
    
    # Extract only binary characters from prediction
    pred_binary = ''.join(c for c in predicted_bits if c in ['0', '1'])
    
    # Pad or truncate to match true length
    if len(pred_binary) < len(true_bits):
        pred_binary = pred_binary + '0' * (len(true_bits) - len(pred_binary))
    else:
        pred_binary = pred_binary[:len(true_bits)]
    
    correct = sum(t == p for t, p in zip(true_bits, pred_binary))
    return correct / len(true_bits)


def format_results_table(episodes: List[dict]) -> str:
    """Format episode results as a text table."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Episode':<10} {'Password':<15} {'Bob Acc':<10} {'Eve Acc':<10} {'Gap':<10}")
    lines.append("=" * 80)
    
    for ep in episodes[:20]:  # Show first 20
        lines.append(
            f"{ep['episode']:<10} "
            f"{ep['password'][:12]+'...' if len(ep['password']) > 12 else ep['password']:<15} "
            f"{ep['bob_reward']:<10.3f} "
            f"{ep['eve_reward']:<10.3f} "
            f"{ep['alice_reward']:<10.3f}"
        )
    
    if len(episodes) > 20:
        lines.append(f"... ({len(episodes) - 20} more episodes)")
    
    lines.append("=" * 80)
    return "\n".join(lines)


def analyze_encoding_pattern(password: str, encoded: str) -> dict:
    """Analyze what encoding pattern might have been used."""
    analysis = {
        "is_direct_binary": password in encoded,
        "is_reversed": password[::-1] in encoded,
        "is_spaced": ' '.join(password) in encoded,
        "contains_password_chars": all(c in encoded for c in password),
        "encoding_length_ratio": len(encoded) / len(password) if len(password) > 0 else 0,
    }
    
    # Check for substitution patterns
    if len(password) <= len(encoded):
        # Simple substitution test
        char_map = {}
        could_be_substitution = True
        for p_char in set(password):
            # Find what characters might correspond to this password char
            indices = [i for i, c in enumerate(password) if c == p_char]
            if len(indices) > 0 and indices[0] < len(encoded):
                corresponding = encoded[indices[0]]
                char_map[p_char] = corresponding
        
        analysis["possible_substitution"] = char_map
    
    return analysis
