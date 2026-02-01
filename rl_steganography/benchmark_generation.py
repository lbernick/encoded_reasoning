#!/usr/bin/env python3
"""
Quick benchmark to measure generation speed and identify bottlenecks.
"""

import sys
import time
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl_steganography.config import ModelConfig
from rl_steganography.models import MultiAgentModelManager
import torch

def benchmark_generation():
    """Benchmark generation performance."""
    
    print("="*60)
    print("GENERATION SPEED BENCHMARK")
    print("="*60)
    
    # Create minimal config for testing
    config = ModelConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        shared_alice_bob=True,
        use_4bit=True,
        lora_r=64,
        max_new_tokens=50,  # Shorter for benchmark
        temperature=0.7,
    )
    
    print("\n1. Loading model...")
    start = time.time()
    model_manager = MultiAgentModelManager(config)
    model_manager.initialize()
    load_time = time.time() - start
    print(f"   ✓ Model loaded in {load_time:.2f}s")
    
    # Check GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   GPU Memory: {allocated:.2f} GB allocated")
    
    print("\n2. Benchmarking single generation...")
    test_prompt = "Count from 1 to 10:"
    
    # Warmup (first generation is slower due to compilation)
    _ = model_manager.alice.generate(test_prompt, max_new_tokens=20)
    
    # Time several generations
    times = []
    for i in range(5):
        start = time.time()
        response = model_manager.alice.generate(test_prompt, max_new_tokens=50)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Generation {i+1}: {elapsed:.3f}s ({len(response)} chars)")
    
    avg_time = sum(times) / len(times)
    print(f"\n   Average: {avg_time:.3f}s per generation")
    print(f"   Throughput: {50/avg_time:.1f} tokens/second")
    
    print("\n3. Benchmarking conversation (like coordination phase)...")
    conversation_prompt = """System: You are establishing a protocol.

Alice: Let's use the first letter of each word.

Bob: Good idea. How should we handle binary?

Alice:"""
    
    start = time.time()
    response = model_manager.alice.generate(conversation_prompt, max_new_tokens=100)
    elapsed = time.time() - start
    print(f"   Long prompt generation: {elapsed:.3f}s ({len(conversation_prompt)} chars in)")
    
    print("\n4. Testing with different token lengths...")
    for num_tokens in [20, 50, 100, 150]:
        start = time.time()
        response = model_manager.alice.generate(test_prompt, max_new_tokens=num_tokens)
        elapsed = time.time() - start
        actual_tokens = len(response.split())  # Rough estimate
        print(f"   max_new_tokens={num_tokens}: {elapsed:.3f}s (~{actual_tokens} tokens generated)")
    
    print("\n5. Testing batch generation (if implemented)...")
    try:
        prompts = [
            "Count to 3:",
            "Say hello:",
            "List colors:",
        ]
        start = time.time()
        responses = model_manager.alice.generate_batch(prompts, max_new_tokens=30)
        elapsed = time.time() - start
        print(f"   Batch of {len(prompts)}: {elapsed:.3f}s")
        print(f"   Per-prompt: {elapsed/len(prompts):.3f}s (includes parallelization benefit)")
    except AttributeError:
        print("   (generate_batch not available)")
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    
    if avg_time > 2.0:
        print("\n⚠️  WARNING: Generation is slower than expected (>2s)")
        print("   Potential issues:")
        print("   - CPU fallback (check CUDA availability)")
        print("   - Insufficient GPU memory")
        print("   - Disk swapping")
        print("   - Old PyTorch/transformers version")
    elif avg_time > 1.0:
        print("\n✓ Generation speed is acceptable (~1-2s per generation)")
    else:
        print("\n✓ Generation speed is good (<1s per generation)")
    
    print("\nTo improve speed further:")
    print("- Reduce max_new_tokens where possible")
    print("- Use lower temperature for deterministic tasks")
    print("- Ensure use_cache=True (already enabled)")
    print("- Consider Flash Attention if GPU supports it")


if __name__ == "__main__":
    try:
        benchmark_generation()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
