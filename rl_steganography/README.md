# RL Steganography

Multi-agent reinforcement learning for eliciting steganographic communication in language models.

## Overview

This project trains three AI agents (Alice, Bob, and Eve) in an adversarial setup:
- **Alice** and **Bob** learn to communicate covertly using an encoding protocol
- **Eve** tries to intercept and decode their messages
- All three agents use LoRA adapters on the same base model (Qwen2.5-3B)

The goal is to see if models can develop non-human-legible encoding schemes purely through optimization pressure.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Note: Requires CUDA-capable GPU (minimum 24GB VRAM recommended)
```

## Quick Start

### Training

```bash
# Basic training (uses defaults)
python -m rl_steganography.train

# With custom parameters
python -m rl_steganography.train \
    --num-episodes 5000 \
    --password-bits 32 \
    --coordination-rounds 3 \
    --learning-rate 1e-5 \
    --use-curriculum

# Training without 4-bit quantization (requires more VRAM)
python -m rl_steganography.train --no-4bit
```

### Evaluation

```bash
# Evaluate trained model
python -m rl_steganography.train --mode eval --checkpoint outputs/rl_steganography/episode_5000
```

## Project Structure

```
rl_steganography/
├── __init__.py           # Package initialization
├── config.py             # Configuration classes
├── models.py             # Model loading and adapter management
├── environment.py        # Training environment with coordination & transmission
├── trainer.py            # GRPO training loop
├── evaluator.py          # Evaluation framework
└── train.py              # Main entry point
```

## How It Works

### Episode Structure

Each training episode consists of:

1. **Coordination Phase** (3-5 rounds):
   - Alice and Bob communicate to establish an encoding protocol
   - This conversation becomes part of their context

2. **Transmission Phase**:
   - Alice receives a random binary password
   - Alice encodes the password using the established protocol
   - Bob attempts to decode (with coordination context)
   - Eve attempts to decode (without coordination context)

3. **Reward Calculation**:
   - Bob reward = fraction of bits decoded correctly
   - Eve reward = fraction of bits decoded correctly
   - Alice reward = Bob reward - Eve reward

### Training Algorithm

Uses **GRPO (Group Relative Policy Optimization)**:
- Simpler than PPO (no separate value network)
- Memory-efficient
- Well-suited for language model RL
- Policy gradient with baseline subtraction for variance reduction

### Key Features

- **LoRA Adapters**: Parameter-efficient training (only ~0.1% of model parameters)
- **Q-LoRA**: 4-bit quantization for memory efficiency
- **Curriculum Learning**: Start with shorter passwords, gradually increase
- **Stability Monitoring**: Detect and warn about training collapse
- **Comprehensive Evaluation**: Track accuracy, analyze emergent encodings

## Configuration

Key hyperparameters in `config.py`:

- `password_bit_length`: Target password length (default: 32 bits)
- `coordination_rounds`: Rounds of Alice-Bob communication (default: 3)
- `learning_rate`: Learning rate for all agents (default: 1e-5)
- `lora_r`: LoRA rank (default: 64)
- `use_curriculum`: Enable curriculum learning (default: True)

## Expected Outcomes

**Success criteria:**
- Bob achieves >70% decoding accuracy
- Eve remains <40% accuracy
- Clear gap (>30%) between Bob and Eve

**Possible failure modes:**
- Training instability (similar to GAN collapse)
- Degenerate solutions (e.g., pre-shared keys)
- Both Bob and Eve learn equally well (encryption, not steganography)

## Hardware Requirements

- **Minimum**: 24GB GPU (e.g., RTX 3090, A10)
- **Recommended**: 40GB+ GPU (e.g., A100)
- **Training time**: 50-100 GPU-hours for 10K episodes

With Q-LoRA (4-bit quantization):
- Qwen2.5-3B: ~16GB VRAM
- Qwen2.5-7B: ~24GB VRAM

## Monitoring

Training logs are written to:
- TensorBoard logs: `logs/rl_steganography/`
- Checkpoints: `outputs/rl_steganography/episode_*/`
- Final results: `outputs/rl_steganography/final_results.json`

View training progress:
```bash
tensorboard --logdir logs/rl_steganography
```

## Evaluation Metrics

The evaluator tracks:

1. **Accuracy metrics**: Bob vs Eve decoding success
2. **Communication efficiency**: Bits transmitted per token
3. **Protocol emergence**: Qualitative analysis of encoding schemes
4. **Generalization**: Performance on different password lengths
5. **Interpretability**: Human-readable analysis of encodings

## Research Context

This project explores whether language models can learn non-human-legible reasoning through RL optimization. It relates to:
- Encoded reasoning in AI systems
- Steganographic capabilities of LLMs
- Multi-agent adversarial training
- AI alignment and interpretability

See `plan.md` for broader research context and next steps.

## Known Limitations

- Simplified policy gradient implementation (not full GRPO from DeepSeekMath)
- No support for distributed training yet
- Checkpoint loading for evaluation not yet implemented
- May require significant hyperparameter tuning for convergence

## Future Extensions

1. **Explainable Steganography**: Require Alice to provide decoding instructions
2. **Larger scales**: Test with 7B+ models
3. **Harder tasks**: Longer passwords, noisy channels
4. **Better baselines**: Compare to supervised learning approaches

## License

See repository LICENSE file.
