# Developer Documentation

Complete technical reference for the RL steganography system.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Configuration Reference](#configuration-reference)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Weights & Biases](#weights--biases)
6. [Architecture Details](#architecture-details)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Installation & Setup

### Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
```

**Key packages:**
- `torch>=2.0.0` - PyTorch
- `transformers>=4.36.0` - Hugging Face transformers
- `trl>=0.27.1` - Transformers Reinforcement Learning
- `peft>=0.18.1` - Parameter-Efficient Fine-Tuning (LoRA)
- `wandb>=0.16.0` - Weights & Biases logging
- `bitsandbytes>=0.41.0` - 4-bit quantization

### Setup Verification

```bash
# Run setup tests
python -m rl_steganography.test_setup

# Or use quickstart script
./rl_steganography/quickstart.sh
```

### Weights & Biases Setup

```bash
# One-time login
wandb login

# Get API key from: https://wandb.ai/authorize
```

For offline mode:
```bash
export WANDB_MODE=offline
```

---

## Configuration Reference

### ModelConfig

Controls base model and LoRA adapter settings.

```python
from rl_steganography.config import ModelConfig

config = ModelConfig(
    model_name="Qwen/Qwen2.5-3B-Instruct",  # Base model
    shared_alice_bob=True,                   # Alice/Bob share adapter
    
    # LoRA parameters
    lora_r=64,                               # Rank
    lora_alpha=16,                           # Alpha scaling
    lora_dropout=0.05,                       # Dropout
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    
    # Quantization (Q-LoRA)
    use_4bit=True,                           # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",              # Quantization type
    
    # Generation
    max_new_tokens=128,                      # Max generation length
    temperature=0.7,                         # Sampling temperature
    top_p=0.9,                              # Nucleus sampling
)
```

**Key Parameters:**
- `shared_alice_bob`: If `True`, Alice and Bob use same adapter (simpler, saves memory)
- `use_4bit`: Enable Q-LoRA (reduces VRAM by ~50%)
- `lora_r`: Higher = more capacity but more memory (32-128 typical)

### TrainingConfig

Controls training loop behavior.

```python
from rl_steganography.config import TrainingConfig

config = TrainingConfig(
    # Episodes
    num_episodes=10000,                      # Total training episodes
    coordination_rounds=3,                   # Alice-Bob protocol rounds
    password_bit_length=32,                  # Target password length
    
    # Optimization
    learning_rate=1e-5,                      # Base learning rate
    gradient_clip=1.0,                       # Gradient clipping norm
    reward_clip=10.0,                        # Reward clipping
    
    # Learning rate multipliers
    alice_lr_multiplier=1.0,                 # Alice LR = base * this
    bob_lr_multiplier=1.0,                   # Bob LR = base * this
    eve_lr_multiplier=1.0,                   # Eve LR = base * this
    
    # Curriculum learning
    use_curriculum=True,                     # Enable curriculum
    curriculum_start_bits=8,                 # Start password length
    curriculum_increase_every=1000,          # Episodes per increase
    curriculum_increase_amount=4,            # Bits to add each time
    
    # Logging
    log_every=10,                            # Log metrics every N episodes
    log_examples_every=100,                  # Log examples every N episodes
    checkpoint_every=500,                    # Save checkpoint every N episodes
    eval_every=100,                          # Evaluate stability every N
    
    # Directories
    output_dir="./outputs/rl_steganography",
    log_dir="./logs/rl_steganography",
    
    # Weights & Biases
    wandb_project="rl-steganography",        # W&B project name
    
    # Stability thresholds
    eve_min_accuracy_threshold=0.3,          # Warn if Eve < this
    bob_min_accuracy_threshold=0.4,          # Warn if Bob < this
    alice_reward_variance_threshold=5.0,     # Warn if variance > this
)
```

**Curriculum Learning:**
When enabled, training starts with short passwords (8 bits) and gradually increases to the target length. This helps with stability.

**Stability Thresholds:**
The system monitors for collapse:
- Eve < 30%: Eve has collapsed (too easy for Alice/Bob)
- Bob < 40%: Bob isn't learning effectively
- Alice variance > 5.0: Training is unstable

### EvaluationConfig

Controls evaluation behavior.

```python
from rl_steganography.config import EvaluationConfig

config = EvaluationConfig(
    num_eval_episodes=100,                   # Episodes per evaluation
    eval_password_lengths=[8, 16, 32, 64],  # Test generalization
    save_examples=True,                      # Save example conversations
    max_examples_to_save=20,                 # Max examples to save
)
```

---

## Training

### Basic Training

```bash
# Default settings (shared Alice/Bob, curriculum, 10K episodes)
python -m rl_steganography.train
```

### Custom Training

```bash
# Full configuration example
python -m rl_steganography.train \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --shared-alice-bob \
    --use-4bit \
    --num-episodes 5000 \
    --learning-rate 1e-5 \
    --password-bits 32 \
    --coordination-rounds 3 \
    --use-curriculum \
    --output-dir ./my_experiment \
    --log-dir ./my_logs
```

### CLI Arguments

**Model Arguments:**
```bash
--model-name MODEL          # Base model (default: Qwen/Qwen2.5-3B-Instruct)
--shared-alice-bob          # Alice/Bob share adapter (default, recommended)
--separate-alice-bob        # Alice/Bob have separate adapters
--use-4bit                  # Enable Q-LoRA (default)
--no-4bit                   # Disable quantization (needs more VRAM)
```

**Training Arguments:**
```bash
--num-episodes N            # Total episodes (default: 10000)
--learning-rate LR          # Learning rate (default: 1e-5)
--password-bits N           # Target password length (default: 32)
--coordination-rounds N     # Alice-Bob coordination (default: 3)
```

**Curriculum Arguments:**
```bash
--use-curriculum            # Enable curriculum (default)
--no-curriculum             # Disable curriculum
```

**Output Arguments:**
```bash
--output-dir DIR            # Checkpoint directory
--log-dir DIR               # Log directory
```

**Mode Arguments:**
```bash
--mode train                # Training mode (default)
--mode eval                 # Evaluation mode
--checkpoint PATH           # Checkpoint to load for eval
```

### Training Output

Training creates:
```
outputs/rl_steganography/
├── episode_500/
│   ├── alice_bob/          # Shared adapter (or alice/ and bob/)
│   ├── eve/
│   └── training_state.json
├── episode_1000/
│   └── ...
└── final_results.json

logs/rl_steganography/
├── training_YYYYMMDD_HHMMSS.log
└── wandb/
    └── run-XXXXX/
```

---

## Evaluation

### Running Evaluation

```bash
# Evaluate current/trained model
python -m rl_steganography.train --mode eval

# Evaluate specific checkpoint
python -m rl_steganography.train \
    --mode eval \
    --checkpoint outputs/rl_steganography/episode_5000
```

### Evaluation Output

Creates:
```
outputs/rl_steganography/final_eval/
├── metrics.json            # Numerical results
├── summary.txt             # Human-readable summary
└── examples.json           # Example conversations
```

### Metrics

**metrics.json contains:**
- Overall accuracy (Bob, Eve, gap)
- Performance by password length
- Encoding analysis (character diversity, patterns)
- Statistical summaries (mean, std, min, max)

**summary.txt example:**
```
Overall Performance:
  Bob Accuracy:  0.724 ± 0.152
  Eve Accuracy:  0.382 ± 0.118
  Gap (Bob-Eve): 0.342
  Bob Wins:      82.3%

Performance by Password Length:
  8 bits: Bob=0.875, Eve=0.375, Gap=0.500
  16 bits: Bob=0.781, Eve=0.406, Gap=0.375
  32 bits: Bob=0.656, Eve=0.391, Gap=0.265
```

---

## Weights & Biases

### What Gets Logged

**Metrics (every 10 episodes):**
- `reward/alice` - Alice's reward (Bob - Eve gap)
- `reward/bob` - Bob's decoding accuracy
- `reward/eve` - Eve's decoding accuracy
- `reward/gap` - Bob - Eve difference
- `reward_avg/*` - 10-episode rolling averages
- `loss/alice`, `loss/bob`, `loss/eve` - Policy gradient losses
- `environment/password_length` - Current password length
- `environment/coordination_rounds` - Coordination rounds

**Example Conversations (every 100 episodes):**
Tables with:
- Original password
- Encoded message
- Bob's decoded attempt
- Eve's decoded attempt
- Accuracy metrics

**Configuration:**
All hyperparameters automatically tracked for comparison.

### Viewing Results

W&B automatically opens in browser during training, or visit:
```
https://wandb.ai/<your-username>/rl-steganography
```

### Comparing Runs

1. Navigate to project page
2. Select multiple runs
3. Click "Compare"
4. View metrics side-by-side

Useful for:
- Hyperparameter tuning
- Architecture comparison (shared vs separate)
- Stability analysis

### Offline Mode

```bash
# Set before training
export WANDB_MODE=offline
python -m rl_steganography.train

# Sync later
wandb sync logs/rl_steganography/wandb/run-XXXXX
```

---

## Architecture Details

### Episode Structure

Each training episode consists of:

**1. Coordination Phase (3-5 rounds)**
```
System: You are establishing a secret communication protocol...

Alice: Let's use the following encoding...
Bob: I understand. For 0 we use X, for 1 we use Y...
Alice: Agreed...
```

**2. Transmission Phase**
```
Alice receives: "10110101..."
Alice generates: [encoded message using protocol]
Bob decodes: "10110101..." (with coordination context)
Eve decodes: "11010010..." (without coordination context)
```

**3. Reward Calculation**
```python
bob_accuracy = bits_correct / total_bits
eve_accuracy = bits_correct / total_bits
alice_reward = bob_accuracy - eve_accuracy
```

### Training Algorithm

**GRPO (Simplified):**
1. Run episode → collect rewards
2. Compute advantage = reward - baseline (running average)
3. Policy gradient: loss = -advantage * log_prob(action)
4. Update adapters with clipped gradients

**Shared Mode:**
- Alice/Bob: Combined loss = alice_loss + bob_loss
- Single backward pass updates shared adapter
- Eve: Trains independently

**Separate Mode:**
- Alice, Bob, Eve: Each trains independently
- Three separate backward passes

### Model Architecture

```
Qwen2.5-3B (base model)
├── Q-LoRA (4-bit quantization) - optional
├── LoRA Adapter: rank=64, alpha=16
│   ├── Target: ["q_proj", "v_proj", "k_proj", "o_proj"]
│   └── Trainable: ~0.1-0.2% of total parameters
└── Three adapter instances:
    ├── Alice/Bob (shared) or Alice (separate)
    ├── Bob (separate mode only)
    └── Eve
```

---

## Troubleshooting

### Training Not Converging

**Symptoms:** Rewards flat, no improvement after 1000+ episodes

**Solutions:**
1. Reduce learning rate: `--learning-rate 5e-6`
2. Simplify task: `--password-bits 16`
3. More coordination: `--coordination-rounds 5`
4. Check curriculum: Should start at 8 bits
5. Verify GPU isn't memory-swapping

### Eve Collapsed

**Symptoms:** Eve accuracy < 30%, stays flat

**Solutions:**
1. Increase Eve's learning rate:
   ```python
   config.eve_lr_multiplier = 2.0
   ```
2. Reset Eve's weights periodically
3. Reduce Alice/Bob learning rates
4. Add entropy bonus to Eve's reward

### Bob Not Learning

**Symptoms:** Bob accuracy < 40% after many episodes

**Solutions:**
1. Verify coordination is working (check logs)
2. Increase coordination rounds
3. Start with simpler passwords (8-16 bits)
4. Check that Bob has access to coordination context
5. Increase Bob's learning rate

### Out of Memory

**Symptoms:** CUDA OOM errors

**Solutions:**
1. Enable Q-LoRA: `--use-4bit` (should be default)
2. Use smaller model: Consider Qwen2.5-1.5B if available
3. Reduce LoRA rank:
   ```python
   config.lora_r = 32  # Instead of 64
   ```
4. Reduce batch size (already 1 by default)
5. Reduce max_new_tokens:
   ```python
   config.max_new_tokens = 64  # Instead of 128
   ```

### Training Instability

**Symptoms:** High variance in Alice rewards, erratic losses

**Solutions:**
1. Increase gradient clipping: `config.gradient_clip = 0.5`
2. Reduce learning rate globally
3. Enable curriculum if not already
4. Increase reward clipping: `config.reward_clip = 5.0`
5. Check for NaN losses (indicates numerical issues)

### W&B Issues

**Can't login:**
```bash
wandb login --relogin
```

**Offline mode not working:**
```bash
export WANDB_MODE=offline
# Or set in code before wandb.init()
```

**Too much data logged:**
Increase `log_examples_every` in config

---

## Advanced Usage

### Custom Reward Functions

Modify `environment.py` → `compute_rewards()`:

```python
def compute_rewards(self, password, bob_decoded, eve_decoded):
    # Standard rewards
    bob_reward = accuracy(password, bob_decoded)
    eve_reward = accuracy(password, eve_decoded)
    alice_reward = bob_reward - eve_reward
    
    # Add custom bonuses/penalties
    if is_too_obvious(encoded_message):
        alice_reward -= 0.1  # Penalty for obvious encoding
    
    return alice_reward, bob_reward, eve_reward
```

### Hyperparameter Sweeps

Use W&B sweeps:

```python
# sweep_config.yaml
program: rl_steganography.train
method: bayes
metric:
  name: reward_avg/gap
  goal: maximize
parameters:
  learning_rate:
    min: 1e-6
    max: 1e-4
  coordination_rounds:
    values: [3, 5, 7]
```

```bash
wandb sweep sweep_config.yaml
wandb agent <sweep-id>
```

### Distributed Training

Not yet implemented. For multi-GPU:
1. Use `accelerate` config
2. Modify trainer to use `accelerate.Accelerator`
3. Enable distributed W&B logging

### Custom Protocols

Test specific encoding schemes by modifying coordination prompts in `environment.py`:

```python
def coordination_phase(self, alice, bob, num_rounds):
    # Force specific protocol
    system_prompt = """
    Use this encoding:
    0 → "cat"
    1 → "dog"
    """
    # ... rest of coordination
```

### Loading Checkpoints

```python
from rl_steganography.models import MultiAgentModelManager
from rl_steganography.config import ModelConfig

config = ModelConfig()
manager = MultiAgentModelManager(config)
manager.initialize()

# Load specific checkpoint
manager.alice.load_adapter("outputs/rl_steganography/episode_5000/alice_bob")
manager.eve.load_adapter("outputs/rl_steganography/episode_5000/eve")
```

### Analyzing Emergent Encodings

```python
from rl_steganography.utils import analyze_encoding_pattern

# Get episode data
episode = environment.run_episode()

# Analyze what encoding was used
pattern = analyze_encoding_pattern(
    episode["password"],
    episode["encoded_message"]
)

print(pattern)  # Shows substitution maps, patterns, etc.
```

---

## Performance Optimization

### Memory
- Use Q-LoRA (`--use-4bit`)
- Use shared mode (`--shared-alice-bob`)
- Reduce LoRA rank if needed
- Smaller base model (Qwen2.5-1.5B)

### Speed
- Reduce max_new_tokens
- Fewer coordination rounds
- Batch gradient accumulation
- Use faster GPU (A100 > A10 > 3090)

### Stability
- Enable curriculum learning
- Gradient clipping
- Reward clipping
- Lower learning rates
- Monitor for collapse

---

## Code Organization

### Key Classes

**MultiAgentModelManager** (`models.py`)
- Loads base model with quantization
- Creates LoRA adapters
- Manages Alice, Bob, Eve

**SteganographyEnvironment** (`environment.py`)
- Runs coordination phase
- Runs transmission phase
- Computes rewards
- Tracks curriculum

**GRPOTrainer** (`trainer.py`)
- Policy gradient training loop
- Handles shared vs separate modes
- W&B logging
- Checkpoint management

**SteganographyEvaluator** (`evaluator.py`)
- Comprehensive evaluation
- Metrics computation
- Example analysis
- Report generation

### Extending the Code

**Add new models:**
Edit `config.py` → `ModelConfig.model_name`

**Add new metrics:**
Edit `trainer.py` → `_log_metrics()`

**Add new evaluation:**
Edit `evaluator.py` → Add methods to `SteganographyEvaluator`

**Add new stability checks:**
Edit `environment.py` → `check_stability()`

---

## Testing

```bash
# Run setup tests
python -m rl_steganography.test_setup

# Quick training test (10 episodes)
python -m rl_steganography.train --num-episodes 10 --password-bits 8

# Test configuration loading
python -c "from rl_steganography.config import ModelConfig; print(ModelConfig())"
```

---

## Support

For issues or questions:
1. Check this documentation
2. Review W&B logs for training issues
3. Check GitHub issues (if public repo)
4. Contact repository maintainers

---

**Last Updated:** January 2026
