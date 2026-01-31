# RL Steganography

Multi-agent reinforcement learning for eliciting steganographic communication in language models.

## Overview

This project trains three AI agents (Alice, Bob, and Eve) in an adversarial setup to test whether language models can develop covert communication protocols through optimization pressure alone.

**The Setup:**
- **Alice** and **Bob** (same or separate adapters) learn to communicate covertly
- **Eve** (adversarial adapter) tries to intercept and decode their messages
- All agents use LoRA adapters on a shared base model (Qwen2.5-3B)

**The Goal:** Determine if models can learn non-human-legible reasoning/encoding schemes purely from RL optimization, without explicit instruction.

## Research Motivation

Standard approaches to testing encoded reasoning rely on prompting ("please reason using emojis"). This project takes a different approach: create genuine optimization pressure where encoding emerges as the optimal strategy.

**Key Questions:**
1. Can models develop steganographic protocols through RL?
2. Do encodings remain human-interpretable or become opaque?
3. What's the performance gap between coordinated parties (Alice/Bob) and adversaries (Eve)?
4. How does this inform AI safety and interpretability?

## Architecture

```
Base Model (Qwen2.5-3B + Q-LoRA)
    ├── Alice Adapter (encoder)
    ├── Bob Adapter (decoder) 
    └── Eve Adapter (adversary)

Episode Flow:
1. Coordination: Alice ↔ Bob establish protocol (3-5 rounds)
2. Transmission: Alice encodes password → Bob & Eve decode
3. Rewards: Alice = Bob_accuracy - Eve_accuracy
           Bob = decoding_accuracy
           Eve = decoding_accuracy
```

**Design Choice:** Alice and Bob share the same adapter by default (simpler, more efficient). Can use separate adapters with `--separate-alice-bob`.

## Quick Start

### Installation

```bash
# Clone/navigate to repo
cd rl_steganography

# Install dependencies
pip install -r requirements.txt

# Login to Weights & Biases (for tracking)
wandb login
```

### Basic Training

```bash
# Quick test (100 episodes, 8-bit passwords)
python -m rl_steganography.train \
    --num-episodes 100 \
    --password-bits 8

# Full training (10K episodes, curriculum learning)
python -m rl_steganography.train

# With separate Alice/Bob adapters
python -m rl_steganography.train --separate-alice-bob
```

### Monitoring

Training automatically logs to Weights & Biases and opens in your browser:
- **Rewards**: Alice, Bob, Eve over time
- **Gap**: Bob - Eve accuracy (success metric)
- **Examples**: Sample conversations every 100 episodes
- **Config**: All hyperparameters tracked

Visit: `https://wandb.ai/<your-username>/rl-steganography`

## Project Structure

```
rl_steganography/
├── __init__.py           # Package init
├── config.py             # ModelConfig, TrainingConfig, EvaluationConfig
├── models.py             # MultiAgentModelManager (LoRA adapters)
├── environment.py        # Episode structure (coordination + transmission)
├── trainer.py            # GRPO training loop
├── evaluator.py          # Evaluation framework
├── train.py              # CLI entry point
├── utils.py              # Helper functions
├── test_setup.py         # Setup verification
└── quickstart.sh         # Automated setup script
```

## Key Features

### Memory Efficient
- **Q-LoRA**: 4-bit quantization (~16GB VRAM for 3B model)
- **LoRA adapters**: Only ~0.1-0.2% of parameters trainable
- **Shared mode**: 2 adapters vs 3 (33% memory savings)

### Training Stability
- **Curriculum learning**: Start with 8-bit passwords, scale to 32
- **Gradient clipping**: Prevent explosions
- **Baseline subtraction**: Reduce variance
- **Collapse detection**: Automatic warnings

### Rich Logging
- **W&B integration**: Interactive dashboards
- **Example tracking**: See emergent encodings
- **Comparison tools**: Side-by-side runs
- **Config management**: Full reproducibility

## Expected Results

### Success Criteria
- Bob accuracy: >70%
- Eve accuracy: <40%
- Gap: >30%
- Training converges within 10K episodes

### Likely Outcomes

**Scenario 1: Full Success (30% probability)**
- Clear steganographic protocols emerge
- Bob significantly outperforms Eve
- Encodings are non-trivial to humans

**Scenario 2: Partial Success (50% probability)**
- Bob learns encoding/decoding
- Eve also improves (encryption vs steganography)
- Gap exists but smaller than target (15-25%)

**Scenario 3: Training Issues (20% probability)**
- Instability prevents convergence
- Requires hyperparameter tuning
- May need curriculum adjustments

## Hardware Requirements

**Minimum:**
- GPU: 24GB VRAM (RTX 3090, A10)
- RAM: 32GB
- Storage: 50GB

**Recommended:**
- GPU: 40GB+ VRAM (A100)
- RAM: 64GB
- Storage: 100GB

**Training Time:**
- 100 episodes (test): ~15-20 minutes
- 10K episodes (full): 50-100 GPU-hours
- Per episode: ~10 seconds

## Configuration Options

```bash
# Model
--model-name Qwen/Qwen2.5-3B-Instruct  # Base model
--shared-alice-bob                      # Alice/Bob share adapter (default)
--separate-alice-bob                    # Separate adapters
--use-4bit / --no-4bit                 # Q-LoRA quantization

# Training
--num-episodes 10000                    # Total episodes
--learning-rate 1e-5                    # Base learning rate
--password-bits 32                      # Target password length
--coordination-rounds 3                 # Alice-Bob coordination

# Curriculum
--use-curriculum / --no-curriculum     # Progressive difficulty
```

See `DEVELOPER.md` for complete configuration reference.

## Research Context

This project explores **Experiment #2** from the broader research plan:
> "Can we finetune a model to reason in an encoded way?"

**Advantages of RL approach:**
- Genuine optimization pressure (not just prompting)
- Bypasses safety filters (supervised finetuning was rejected)
- Tests emergent behavior under constraints

**Next Steps:**
1. **Explainable steganography**: Alice must provide decoding instructions
2. **Evolving adversaries**: Multiple Eves that improve over time
3. **Information theory**: Mutual information constraints
4. **SAE analysis**: Check if encoded reasoning activates same concepts

## Documentation

- **This file**: Project overview and quick start
- **DEVELOPER.md**: Technical details, configuration, troubleshooting
- **Code**: Inline docstrings throughout