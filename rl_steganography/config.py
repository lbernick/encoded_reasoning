"""Configuration for RL steganography training."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for base model and adapters."""
    
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    model_revision: Optional[str] = None
    
    # Agent configuration
    shared_alice_bob: bool = True  # If True, Alice and Bob share the same adapter
    
    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = None
    
    # Quantization
    use_4bit: bool = True  # Q-LoRA
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    
    # Episode configuration
    num_episodes: int = 10000
    coordination_rounds: int = 3
    password_bit_length: int = 32
    
    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    
    # GRPO parameters
    grpo_clip_range: float = 0.2
    grpo_value_clip_range: float = 0.2
    
    # Stability parameters
    reward_clip: float = 10.0
    gradient_clip: float = 1.0
    
    # Adaptive learning rates
    alice_lr_multiplier: float = 1.0
    bob_lr_multiplier: float = 1.0
    eve_lr_multiplier: float = 1.0
    
    # Checkpointing
    checkpoint_every: int = 500
    eval_every: int = 100
    log_every: int = 10
    
    # Output directories
    output_dir: str = "./outputs/rl_steganography"
    log_dir: str = "./logs/rl_steganography"
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_start_bits: int = 8
    curriculum_increase_every: int = 1000
    curriculum_increase_amount: int = 4
    
    # Anti-collapse mechanisms
    eve_min_accuracy_threshold: float = 0.3  # Reset if Eve drops below this
    bob_min_accuracy_threshold: float = 0.4  # Warning if Bob below this
    alice_reward_variance_threshold: float = 5.0  # Warning if variance too high
    
    # Weights & Biases
    wandb_project: str = "rl-steganography"
    log_examples_every: int = 100  # Log example conversations to W&B


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    num_eval_episodes: int = 100
    eval_password_lengths: list[int] = None
    save_examples: bool = True
    max_examples_to_save: int = 20
    
    def __post_init__(self):
        if self.eval_password_lengths is None:
            self.eval_password_lengths = [8, 16, 32, 64]
