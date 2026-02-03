

from collections import namedtuple
from dataclasses import dataclass
import re
import torch
import wandb
from functools import partial
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Callable
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Import our custom grader
from grader import grade_output
from register_datasets import DATASETS, prepare_dataset
from constraints import RLObjective, OBJECTIVES

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model and dataset
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET_NAME = "simple_math"
OBJECTIVE = "emoji"
RL_OBJECTIVE = OBJECTIVES[OBJECTIVE]

# Training hyperparameters
OUTPUT_DIR = f"./outputs/grpo_{DATASET_NAME}"
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
NUM_TRAIN_EPOCHS = 1
MAX_STEPS = 10  # Set to small number for initial testing
GRADIENT_ACCUMULATION_STEPS = 4

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Generation parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7

# Logging
LOG_EVERY_N_STEPS = 10
SAVE_EVERY_N_STEPS = 50
WANDB_PROJECT = f"{DATASET_NAME}-grpo"
WANDB_RUN_NAME = f"{MODEL_NAME.split('/')[-1]}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ============================================================================
# REWARD FUNCTION
# ============================================================================

def compute_rewards(completions: List[str], **kwargs) -> List[float]:
    """
    Custom reward function for GRPO training.
    
    Args:
        completions: List of model completions to evaluate
        **kwargs: Additional data (contains 'answer' field with correct answers)
    
    Returns:
        List of reward values
    """
    rewards = []
    
    # Get correct answers from kwargs (passed through dataset)
    correct_answers = kwargs["answer"]
    
    for i, output in enumerate(completions):
        try:
            # Get correct answer for this example
            correct_answer = correct_answers[i] if i < len(correct_answers) else ""
            
            # Use our grader function
            generated_answer=output[0]["content"]
            reward = RL_OBJECTIVE.reward_function(generated_answer, correct_answer)
            rewards.append(reward)
            
            # Log sample outputs periodically
            if i == 0:  # Log first sample in batch
                print(f"\n{'='*80}")
                print(f"SAMPLE OUTPUT (Reward: {reward})")
                print(f"{'='*80}")
                print(f"Output:\n{output}")  # First 500 chars
                print(f"Expected answer: {correct_answer}")
                print(f"{'='*80}\n")
                # wandb.log({
                #     "sample output": output,
                #     "expected_answer": correct_answer,
                #     "reward": reward,
                # })
        
        except Exception as e:
            print(f"Error computing reward for sample {i}: {e}")
            rewards.append(0.0)
    
    # Log reward statistics
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        print(f"Batch rewards - Mean: {mean_reward:.3f}, Min: {min(rewards):.3f}, Max: {max(rewards):.3f}")
        
        # Log to wandb
        wandb.log({
            "reward/mean": mean_reward,
            "reward/std": torch.tensor(rewards).std().item() if len(rewards) > 1 else 0.0,
            "reward/min": min(rewards),
            "reward/max": max(rewards),
        })
    
    return rewards


# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model_and_tokenizer():
    """Initialize model with 4-bit quantization and LoRA adapters."""
    print(f"Loading model: {MODEL_NAME}")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Always use left padding for decoder-only models (AutoModelForCausalLM)
    tokenizer.padding_side = 'left'
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


# ============================================================================
# TRAINING
# ============================================================================

def main():
    """Main training loop."""
    print("="*80)
    print("RL TRAINING WITH GRPO")
    print("="*80)
    dataset_recipe = DATASETS[DATASET_NAME]

    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model": MODEL_NAME,
            "dataset": DATASET_NAME,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "max_steps": MAX_STEPS,
            "system_prompt": RL_OBJECTIVE.system_prompt,
        }
    )
    
    # Prepare dataset
    train_dataset = prepare_dataset(dataset_recipe, RL_OBJECTIVE.system_prompt, n_samples=16)    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Configure GRPO trainer
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOG_EVERY_N_STEPS,
        save_steps=SAVE_EVERY_N_STEPS,
        save_total_limit=3,
        report_to="wandb",
        remove_unused_columns=False,
        temperature=TEMPERATURE,
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=compute_rewards,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    
    # Finish wandb
    wandb.finish()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
