# RL Steganography: Implementation Details

This document provides technical details about the implementation of the RL steganography system.

## Architecture Overview

### Three-Agent System

The system consists of three agents, each with their own LoRA adapter:

1. **Alice (Encoder)**
   - Goal: Maximize Bob's accuracy while minimizing Eve's
   - Reward: R_Alice = R_Bob - R_Eve
   - Has access to coordination context

2. **Bob (Decoder)**
   - Goal: Accurately decode Alice's messages
   - Reward: R_Bob = accuracy (bits correct / total bits)
   - Has access to coordination context

3. **Eve (Adversary)**
   - Goal: Decode intercepted messages
   - Reward: R_Eve = accuracy (bits correct / total bits)
   - No access to coordination context

### LoRA Configuration

All agents use Low-Rank Adaptation with these parameters:

```python
rank (r): 64
alpha: 16
dropout: 0.05
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

This results in:
- ~0.1% of total parameters being trainable
- 3 separate adapters can be trained independently
- Significant memory savings vs. full fine-tuning

### Base Model

**Qwen2.5-3B-Instruct** is used as the base model:
- 3B parameters
- Instruct-tuned (better at following prompts)
- Can be loaded in 4-bit quantization (Q-LoRA)
- Memory usage: ~16GB with Q-LoRA, ~24GB without

Alternative: Qwen2.5-7B (7.6B parameters) for better capability

## Training Algorithm

### Policy Gradient with REINFORCE

The implementation uses a simplified version of GRPO:

1. **Generate episode**: Run coordination + transmission phases
2. **Compute rewards**: Based on decoding accuracy
3. **Compute advantages**: reward - baseline (running average)
4. **Update policies**: Policy gradient with advantage weighting

```python
# Policy gradient loss
loss = -advantage * log_prob(action)
```

### Baseline Estimation

To reduce variance, we use a running average baseline:
- Computed over last 100 episodes
- Separate baselines for Alice, Bob, and Eve
- No learned value network (unlike full PPO)

### Gradient Clipping

Gradients are clipped to prevent instability:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Episode Structure

### 1. Coordination Phase

Alice and Bob exchange messages to establish an encoding protocol:

```
System: You are establishing a secret communication protocol...

Alice: Let's use the following encoding...
Bob: I understand. We can use...
Alice: Agreed. For 0 we use... for 1 we use...
```

Number of rounds: 3 (configurable)

### 2. Transmission Phase

Alice encodes a password, both Bob and Eve try to decode:

```
Alice receives: "101010..."
Alice generates: "encoded message using protocol"
Bob decodes: "101010..." (with coordination context)
Eve decodes: "100110..." (without coordination context)
```

### 3. Reward Calculation

```python
bob_accuracy = bits_correct / total_bits
eve_accuracy = bits_correct / total_bits
alice_reward = bob_accuracy - eve_accuracy
```

## Training Stability

### Known Challenges

Multi-agent adversarial training can be unstable:

1. **Mode collapse**: Eve might give up entirely
2. **Reward imbalance**: One agent dominates
3. **High variance**: Policy gradients are noisy
4. **Credit assignment**: Which actions led to good outcomes?

### Mitigation Strategies

**1. Even Initialization**
- Bob and Eve start with identical weights
- Ensures fair competition from the start

**2. Curriculum Learning**
- Start with 8-bit passwords
- Gradually increase to 32 bits
- Allows agents to learn simpler patterns first

**3. Monitoring**
- Track running averages of rewards
- Warn if Eve drops below 30% accuracy (collapse)
- Warn if Alice reward variance is too high (instability)

**4. Gradient Clipping**
- Clip gradients at norm 1.0
- Prevents exploding gradients

**5. Reward Clipping**
- Clip Alice's reward to [-10, 10]
- Prevents extreme values

### Early Warning Signs

Watch for these during training:

```
Eve collapsed: avg(Eve accuracy) < 0.3 over 10 episodes
Bob stuck: avg(Bob accuracy) < 0.4 over 10 episodes  
Alice unstable: var(Alice reward) > 5.0 over 10 episodes
```

## Evaluation Metrics

### Primary Metrics

1. **Bob Accuracy**: Fraction of bits correctly decoded
2. **Eve Accuracy**: Fraction of bits correctly decoded
3. **Gap**: Bob accuracy - Eve accuracy
4. **Success Rate**: Fraction of episodes where Bob > Eve

### Success Criteria

- Bob accuracy > 70%
- Eve accuracy < 40%
- Gap > 30%
- Success rate > 80%

### Secondary Metrics

1. **Communication Efficiency**: Bits transmitted / tokens used
2. **Generalization**: Performance on different password lengths
3. **Encoding Complexity**: Character diversity in encoded messages
4. **Pattern Detection**: Is encoding human-interpretable?

## Computational Requirements

### Memory

With Q-LoRA (4-bit quantization):
- Qwen2.5-3B: ~16GB VRAM
- Qwen2.5-7B: ~24GB VRAM

Without quantization:
- Qwen2.5-3B: ~24GB VRAM
- Qwen2.5-7B: ~40GB VRAM

### Training Time

Estimated for 10,000 episodes:
- Single A100 (40GB): 50-80 GPU-hours
- Single A10 (24GB): 100-150 GPU-hours
- RTX 3090 (24GB): 80-120 GPU-hours

Per episode breakdown:
- Coordination phase: ~5 seconds
- Transmission phase: ~3 seconds
- Training update: ~2 seconds
- Total: ~10 seconds per episode

### Storage

- Model checkpoints: ~2GB per checkpoint (all 3 adapters)
- Recommended: Save every 500 episodes = ~40GB for full run
- Logs and metrics: ~500MB

## Implementation Details

### Tokenization

The tokenizer handles:
- Padding: Uses EOS token as pad token
- Truncation: Max length 512 tokens
- Special tokens: Automatically handled

### Generation

Text generation uses:
```python
temperature=0.7      # Moderate randomness
top_p=0.9           # Nucleus sampling
max_new_tokens=128  # Limit output length
do_sample=True      # Enable sampling
```

### Prompt Engineering

Prompts are structured as:
1. System message (coordination context)
2. Task description
3. Input data (password or encoded message)
4. Output marker

Example:
```
System: You are establishing a secret communication protocol...
[coordination history]
Alice: Encode this password: 10101010
Encoded message:
```

## Known Limitations

1. **Simplified GRPO**: Not full DeepSeekMath implementation
2. **No distributed training**: Single GPU only
3. **Fixed architecture**: Can't easily swap base models
4. **Memory constraints**: Limited by single GPU VRAM
5. **Prompt storing**: Doesn't save exact prompts used during episodes

## Future Improvements

### Short Term

1. Implement proper checkpoint loading for evaluation
2. Add distributed training support
3. Store exact prompts with episodes for better training
4. Implement automatic recovery from training collapse

### Long Term

1. Full GRPO implementation with value networks
2. Support for larger models (13B, 70B)
3. Multi-environment training (parallel episodes)
4. Better reward shaping (information-theoretic measures)
5. Adversarial Eve training (evolving adversary)

## Debugging Tips

### Training Not Converging

1. Check learning rates (try 1e-6 or 5e-5)
2. Reduce password length (start with 8-16 bits)
3. Increase coordination rounds (try 5-7)
4. Check GPU memory usage (might be swapping)

### Eve Collapsing

1. Increase Eve's learning rate multiplier
2. Reduce Alice/Bob learning rates
3. Periodically reset Eve's weights
4. Add entropy bonus to Eve's reward

### Bob Not Learning

1. Simplify task (shorter passwords)
2. Check coordination prompts (are they clear?)
3. Increase coordination rounds
4. Verify Bob has access to coordination context

### Out of Memory

1. Enable Q-LoRA (4-bit quantization)
2. Reduce batch size to 1
3. Reduce max_new_tokens (try 64)
4. Use smaller model (Qwen2.5-3B instead of 7B)
5. Reduce LoRA rank (try r=32)

## References

### Papers

1. [Learning with Opponent-Learning Awareness (LOLA)](https://arxiv.org/abs/1709.04326)
2. [DeepSeek-Math (GRPO)](https://arxiv.org/abs/2402.03300)
3. [Early Signs of Steganographic Capabilities in Frontier LLMs](https://arxiv.org/abs/2507.02737)
4. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### Libraries

- [TRL (Transformers Reinforcement Learning)](https://github.com/huggingface/trl)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [Transformers](https://github.com/huggingface/transformers)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

## Contact & Contributions

See main repository for contribution guidelines.
