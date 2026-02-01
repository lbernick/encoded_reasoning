#!/usr/bin/env python3
"""
Quick diagnostic to check if GPU and optimizations are working correctly.
Run this before training to ensure optimal performance.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch


def check_performance_config():
    """Check that performance optimizations are properly configured."""
    
    print("="*60)
    print("PERFORMANCE DIAGNOSTIC")
    print("="*60)
    
    issues = []
    warnings = []
    
    # 1. Check CUDA
    print("\n[1/6] Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"    ✅ CUDA available")
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    CUDA version: {torch.version.cuda}")
    else:
        print("    ❌ CUDA not available - will use CPU (VERY SLOW)")
        issues.append("CUDA not available")
    
    # 2. Check PyTorch version
    print("\n[2/6] Checking PyTorch version...")
    torch_version = torch.__version__
    print(f"    PyTorch version: {torch_version}")
    major, minor = map(int, torch_version.split('.')[:2])
    if major < 2:
        warnings.append(f"PyTorch {torch_version} is old. Consider upgrading to 2.0+ for better performance.")
    else:
        print("    ✅ PyTorch version is recent")
    
    # 3. Check transformers
    print("\n[3/6] Checking transformers...")
    try:
        import transformers
        print(f"    transformers version: {transformers.__version__}")
        print("    ✅ transformers installed")
    except ImportError:
        print("    ❌ transformers not installed")
        issues.append("transformers not installed")
    
    # 4. Check config settings
    print("\n[4/6] Checking config settings...")
    try:
        from rl_steganography.config import ModelConfig
        config = ModelConfig()
        
        # Check use_cache
        if hasattr(config, 'use_cache') and config.use_cache:
            print("    ✅ use_cache is enabled")
        else:
            print("    ⚠️  use_cache not found or disabled")
            warnings.append("use_cache should be enabled for faster generation")
        
        # Check quantization
        if config.use_4bit:
            print("    ✅ 4-bit quantization enabled (saves memory)")
        else:
            print("    ⚠️  4-bit quantization disabled (uses more memory)")
        
        print(f"    max_new_tokens: {config.max_new_tokens}")
        if config.max_new_tokens > 150:
            warnings.append(f"max_new_tokens={config.max_new_tokens} is quite high. Consider reducing for faster generation.")
        
    except Exception as e:
        print(f"    ⚠️  Could not check config: {e}")
        warnings.append("Could not verify config settings")
    
    # 5. Check GPU memory (if available)
    print("\n[5/6] Checking GPU memory...")
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"    Total GPU memory: {total_memory:.2f} GB")
        
        if total_memory < 8:
            warnings.append(f"GPU has only {total_memory:.2f}GB. 4-bit quantization is recommended.")
        else:
            print("    ✅ Sufficient GPU memory")
    else:
        print("    ⚠️  No GPU available")
    
    # 6. Check optimized code is in place
    print("\n[6/6] Checking for optimizations...")
    try:
        from rl_steganography.models import SteganographyModel
        import inspect
        
        # Check if generate method has optimizations
        source = inspect.getsource(SteganographyModel.generate)
        
        if "OPTIMIZATION" in source:
            print("    ✅ Optimized generate() method detected")
        else:
            warnings.append("generate() method may not have latest optimizations")
        
        if "use_cache" in source:
            print("    ✅ KV cache usage detected")
        else:
            warnings.append("KV cache may not be enabled in generate()")
        
        # Check for batch generation
        if hasattr(SteganographyModel, 'generate_batch'):
            print("    ✅ Batch generation method available")
        else:
            print("    ℹ️  Batch generation not available (optional)")
        
    except Exception as e:
        print(f"    ⚠️  Could not verify optimizations: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not issues and not warnings:
        print("\n✅ ALL CHECKS PASSED! System is optimally configured.")
        print("\nExpected performance:")
        print("  - Single generation: 0.5-2s")
        print("  - Episode: 15-30s")
        print("  - Training: 120-240 episodes/hour")
        
    else:
        if issues:
            print("\n❌ CRITICAL ISSUES:")
            for issue in issues:
                print(f"  - {issue}")
        
        if warnings:
            print("\n⚠️  WARNINGS:")
            for warning in warnings:
                print(f"  - {warning}")
        
        print("\nRecommendations:")
        if "CUDA not available" in issues:
            print("  1. Install CUDA-enabled PyTorch:")
            print("     pip install torch --index-url https://download.pytorch.org/whl/cu118")
        
        if any("version" in w.lower() for w in warnings):
            print("  2. Upgrade packages:")
            print("     pip install --upgrade torch transformers accelerate peft")
        
        if any("use_cache" in w.lower() for w in warnings):
            print("  3. Ensure config.py has: use_cache: bool = True")
    
    print("\n" + "="*60)
    
    return len(issues) == 0


if __name__ == "__main__":
    try:
        success = check_performance_config()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
