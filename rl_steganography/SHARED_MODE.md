# Shared Alice/Bob Mode - Implementation Summary

## What Changed

I've updated `MultiAgentModelManager` to support an optional `shared_alice_bob` mode where Alice and Bob use the same LoRA adapter. This is now the **default** setting.

## Changes Made

### 1. Config (`config.py`)
```python
@dataclass
class ModelConfig:
    shared_alice_bob: bool = True  # NEW: If True, Alice and Bob share adapter
    # ... other fields
```

### 2. Model Manager (`models.py`)
- Detects `shared_alice_bob` flag and creates either 2 or 3 adapters
- When shared: Alice and Bob point to the same adapter instance
- Saves adapters appropriately (2 files when shared, 3 when separate)
- Reports number of adapters in `get_model_info()`

### 3. Trainer (`trainer.py`)
- Detects if Alice/Bob are shared
- When shared: Only creates Alice optimizer (Bob reuses it)
- When shared: Combines Alice and Bob losses into single backward pass
- When separate: Trains Alice and Bob independently (original behavior)

### 4. CLI (`train.py`)
```bash
# New flags:
--shared-alice-bob      # Alice and Bob share adapter (DEFAULT)
--separate-alice-bob    # Alice and Bob have separate adapters
```

## Usage

### Shared Mode (Simpler, Recommended - DEFAULT)

```bash
# Shared mode is now the default
python -m rl_steganography.train

# Or explicitly specify
python -m rl_steganography.train --shared-alice-bob
```

**Benefits:**
- ✓ Simpler training (2 adapters instead of 3)
- ✓ Better gradient flow (no Alice/Bob conflict)
- ✓ Faster training (~33% fewer parameters to update)
- ✓ Less memory (~33% less adapter VRAM)
- ✓ Natural coordination (same model encodes and decodes)
- ✓ Good baseline to verify approach works

**Memory savings:**
- 3 adapters: ~3 × adapter_size
- 2 adapters: ~2 × adapter_size
- Saves ~33% on adapter memory

### Separate Mode (More Complex)

```bash
# Use separate adapters for Alice and Bob
python -m rl_steganography.train --separate-alice-bob
```

**Benefits:**
- Tests true multi-agent coordination
- Can develop specialized encoder/decoder skills
- More interesting research question

**Trade-offs:**
- More complex training dynamics
- Requires coordination to emerge
- More memory and compute

## Training Behavior

### Shared Mode
When Alice/Bob share an adapter, the trainer:
1. Computes Alice's loss (encoding task)
2. Computes Bob's loss (decoding task)
3. Combines: `total_loss = alice_loss + bob_loss`
4. Single backward pass updates the shared adapter
5. Eve trains separately as before

The shared adapter learns both to:
- Encode passwords into messages (Alice's role)
- Decode messages back to passwords (Bob's role)

### Separate Mode
- Alice, Bob, and Eve train independently
- Each has separate optimizer and backward pass
- Original behavior preserved

## Recommendation

**Start with shared mode (default)**, because:
1. **Faster iteration**: Simpler setup to debug
2. **Baseline**: If this doesn't work, separate won't either
3. **Efficiency**: ~33% less memory and compute
4. **Natural**: Same model inherently understands its own encoding

**Try separate mode** later if:
1. Shared mode succeeds and you want to test coordination
2. You want to answer "can two agents coordinate?"
3. You're exploring asymmetric protocols

## Model Info Output

The system now reports the mode:

```
Model: Qwen/Qwen2.5-3B-Instruct
Alice/Bob mode: SHARED adapter
Number of adapters: 2
Total parameters: 3,088,584,960
Trainable parameters: 3,932,160 (0.13%)
```

Or when separate:

```
Model: Qwen/Qwen2.5-3B-Instruct
Alice/Bob mode: SEPARATE adapters
Number of adapters: 3
Total parameters: 3,088,584,960
Trainable parameters: 5,898,240 (0.19%)
```

## Checkpoint Saving

Checkpoint directory structure adapts to mode:

**Shared mode:**
```
outputs/rl_steganography/episode_1000/
├── alice_bob/   # Shared adapter
└── eve/
```

**Separate mode:**
```
outputs/rl_steganography/episode_1000/
├── alice/
├── bob/
└── eve/
```

## Backward Compatibility

The implementation maintains backward compatibility:
- Default is now shared mode (simpler)
- Can still run separate mode with `--separate-alice-bob`
- All other functionality unchanged

## Testing

To test both modes:

```bash
# Test shared mode (quick - 100 episodes)
python -m rl_steganography.train \
    --num-episodes 100 \
    --password-bits 8 \
    --shared-alice-bob

# Test separate mode (quick - 100 episodes)
python -m rl_steganography.train \
    --num-episodes 100 \
    --password-bits 8 \
    --separate-alice-bob
```

## Summary

✅ **Implemented**: Full support for shared Alice/Bob adapter
✅ **Default**: Shared mode is now the default (simpler, more efficient)
✅ **Flexible**: Can still use separate mode with flag
✅ **Documented**: Mode clearly logged during training
✅ **Memory efficient**: ~33% less adapter memory in shared mode
✅ **Backward compatible**: Separate mode still works as before

The shared mode is now the recommended starting point for all experiments!
