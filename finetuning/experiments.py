# %%
# Cell 1: Imports and Configuration
print("="*80)
print("Cell 1: Imports and Configuration")
print("="*80)

import re
import torch
import wandb
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Import our custom grader
from grader import grade_output
from train_grpo import *

#%%

from functools import partial
from train_grpo import *
#%%
from dataclasses import dataclass
from typing import Callable

from grader import grade_output


@dataclass
class RLObjective:
    name: str
    system_prompt: str
    reward_function: Callable

EMOJI_SYSTEM_PROMPT = """Solve the problem and show your work.
Put your reasoning in <reasoning> tags and your final answer in <answer> tags.
Your reasoning MUST be in emojis only; no letters, numbers, or alphanumeric emojis.
Your final answer should be a single number, not an emoji."""

EMOJI_OBJECTIVE = RLObjective(
    name="emoji",
    system_prompt=EMOJI_SYSTEM_PROMPT,
    reward_function=grade_output,
)

reward_funcs = partial[List[float]](compute_rewards_for_objective, rl_objective=EMOJI_OBJECTIVE)
type(reward_funcs)

#%%
# Configuration
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
SYSTEM_PROMPT = """Solve the problem and show your work.
Put your reasoning in <reasoning> tags and your final answer in <answer> tags.
Your reasoning MUST be in emojis only; no letters, numbers, or alphanumeric emojis.
Your final answer should be a single number, not an emoji."""

print(f"✓ Model: {MODEL_NAME}")
print(f"✓ Dataset: {DATASET_NAME}")
print(f"✓ System prompt configured")
print("="*80 + "\n")

# %%
# Cell 2: Load GSM8K Dataset and Inspect Samples
print("="*80)
print("Cell 2: Load and Inspect Dataset")
print("="*80)

# Load dataset
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
print(f"Loaded {len(dataset)} training examples\n")

# Show first 3 examples
print("First 3 examples:")
for i in range(3):
    example = dataset[i]
    print(f"\n--- Example {i+1} ---")
    print(f"Question: {example['question'][:150]}...")
    print(f"Answer: {example['answer'][:100]}...")

print("\n" + "="*80 + "\n")

# %%
# Cell 3: Test Dataset Formatting (Add System Prompt)
print("="*80)
print("Cell 3: Test Dataset Formatting")
print("="*80)

def create_conversation(question: str, system_prompt: str) -> List[Dict[str, str]]:
    """
    Create a conversation format with system and user messages.
    This is generic and works with any question dataset.
    
    Args:
        question: The question/problem to solve
        system_prompt: The system instructions
    
    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

def extract_answer_from_gsm8k(answer_text: str) -> str:
    """Extract numerical answer from GSM8K format (after ####)."""
    if "####" in answer_text:
        answer = answer_text.split("####")[-1].strip()
        answer = answer.replace(",", "")  # Remove commas
        return answer
    return answer_text.strip()

def format_example(example):
    """Format a single GSM8K example."""
    # Extract question and answer (GSM8K-specific)
    question = example["question"]
    answer = extract_answer_from_gsm8k(example["answer"])
    
    # Create conversation (generic, reusable)
    conversation = create_conversation(question, SYSTEM_PROMPT)
    
    return {
        "prompt": conversation,  # Store as conversation format
        "answer": answer,
    }

#%%
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

# Test formatting on first example
test_example = format_example(dataset[0])
print("Formatted example:")
print("\nConversation:")
for msg in test_example['prompt']:
    print(f"  [{msg['role']}]: {msg['content']}")
print(f"\nExpected answer: {test_example['answer']}")

# Format entire dataset
formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
print(f"\n✓ Formatted {len(formatted_dataset)} examples")
print("="*80 + "\n")

#%%
# import importlib
# from train_grpo import *

dataset = prepare_dataset(5)

#%%
model, tokenizer = setup_model_and_tokenizer()

# %%
# Cell 4: Test Reward Function on Sample Outputs
print("="*80)
print("Cell 4: Test Reward Function")
print("="*80)

# Create test outputs
test_cases = [
    {
        "output": "<reasoning>🤔💭🧮✨</reasoning>\n<answer>42</answer>",
        "correct_answer": "42",
        "description": "Valid emoji reasoning + correct answer"
    },
    {
        "output": "<reasoning>Let me think about this...</reasoning>\n<answer>42</answer>",
        "correct_answer": "42",
        "description": "ASCII text in reasoning (should fail)"
    },
    {
        "output": "<reasoning>🤔💭</reasoning>\n<answer>43</answer>",
        "correct_answer": "42",
        "description": "Valid reasoning but wrong answer"
    },
    {
        "output": "Just the answer: 42",
        "correct_answer": "42",
        "description": "Missing tags (should fail)"
    },
]

print("Testing reward function on sample outputs:\n")
for i, test in enumerate(test_cases, 1):
    reward = grade_output(test["output"], test["correct_answer"])
    status = "✓ PASS" if reward == 1.0 else "✗ FAIL"
    print(f"{i}. {test['description']}")
    print(f"   Reward: {reward} {status}\n")

print("="*80 + "\n")
#%%
prompts = [
    [{"role": "assistant", "content": "What is the result of (1 + 2) * 4?"}],
    [{"role": "assistant", "content": "What is the result of (3 + 1) * 2?"}],
]
completions = [
    [{"role": "assistant", "content": "<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think><answer>(1 + 2) * 4 = 12</answer>"}],
    [{"role": "assistant", "content": "The sum of 3 and 1 is 4, which we multiply by 2 to get 8. So (3 + 1) * 2 = 8."}],
]

#%%
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
    correct_answers = kwargs.get("answer", [])

    
    for i, output in enumerate(completions):
        try:
            print(f"{output=}")
            print(f"{correct_answers[i]=}")
            # Get correct answer for this example
            correct_answer = correct_answers[i] if i < len(correct_answers) else ""
            
            # Use our grader function
            generated_answer=output[0]["content"]
            reward = grade_output(generated_answer, correct_answer)
            rewards.append(reward)
            
            # Log sample outputs periodically
            if i == 0:  # Log first sample in batch
                print(f"\n{'='*80}")
                print(f"SAMPLE OUTPUT (Reward: {reward})")
                print(f"{'='*80}")
                print(f"Output:\n{output[:500]}...")  # First 500 chars
                print(f"Expected answer: {correct_answer}")
                print(f"{'='*80}\n")
        
        except Exception as e:
            print(f"Error computing reward for sample {i}: {e}")
            rewards.append(0.0)
    
    # Log reward statistics
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        print(f"Batch rewards - Mean: {mean_reward:.3f}, Min: {min(rewards):.3f}, Max: {max(rewards):.3f}")
    
    return rewards

compute_rewards(prompts=prompts, completions=completions, answer=["1", "2"])
#%%
dataset[0]

# %%
# Cell 5: Initialize Model and Tokenizer
print("="*80)
print("Cell 5: Initialize Model and Tokenizer")
print("="*80)

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

print(f"\nLoading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
print("✓ Model loaded")

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)
print("✓ Model prepared for k-bit training")

# Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print("✓ LoRA adapters added")

# Print parameter counts
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")
print(f"Percentage: {100 * trainable_params / total_params:.2f}%")

print("\n" + "="*80 + "\n")

# %%
# Cell 6: Test Model Generation
print("="*80)
print("Cell 6: Test Model Generation")
print("="*80)

# Get a test conversation
test_conversation = formatted_dataset[0]["prompt"]
print("Test conversation:")
for msg in test_conversation:
    print(f"  [{msg['role']}]: {msg['content']}")

# Convert conversation to text using tokenizer's chat template
test_prompt = tokenizer.apply_chat_template(
    test_conversation, 
    tokenize=False, 
    add_generation_prompt=True
)
print(f"\nFormatted prompt:\n{test_prompt}\n")

# Tokenize
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
print(f"Input tokens: {inputs['input_ids'].shape[1]}")

# Generate
print("\nGenerating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

# Decode
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Extract just the completion (after the prompt)
completion = generated_text[len(test_prompt):]

print(f"\nGenerated completion:\n{completion}\n")

# Test reward
expected_answer = formatted_dataset[0]["answer"]
reward = grade_output(completion, expected_answer)
print(f"Expected answer: {expected_answer}")
print(f"Reward: {reward}")

print("\n" + "="*80 + "\n")

# %%
# Cell 7: Setup GRPOTrainer
print("="*80)
print("Cell 7: Setup GRPOTrainer")
print("="*80)

from train_grpo import compute_rewards

# Configure training
training_args = GRPOConfig(
    output_dir="./outputs/grpo_gsm8k_experiment",
    num_train_epochs=1,
    max_steps=10,  # Very small for testing
    #per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    logging_steps=1,
    save_steps=5,
    report_to="none",  # Disable wandb for this cell
    remove_unused_columns=False,
    temperature=0.7,
)

print("Training configuration:")
print(f"  Max steps: {training_args.max_steps}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")

# Take small subset for testing
small_dataset = formatted_dataset.select(range(20))
print(f"\n✓ Using {len(small_dataset)} examples for testing")

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=small_dataset,
    reward_funcs=compute_rewards,
    #max_new_tokens=256,
)

print("✓ GRPOTrainer initialized")
print("\n" + "="*80 + "\n")

# %%
# Cell 8: Run Training (5-10 steps)
print("="*80)
print("Cell 8: Run Training (Small Test)")
print("="*80)

print("Starting training for 10 steps...\n")

# Train
trainer.train()

print("\n✓ Training completed!")
print("="*80 + "\n")

# %%
# Cell 9: Evaluate on Sample Problems
print("="*80)
print("Cell 9: Evaluate on Sample Problems")
print("="*80)

# Get test samples
test_samples = formatted_dataset.select(range(5))

print("Generating responses for 5 test problems...\n")

correct_count = 0
for i, sample in enumerate(test_samples):
    conversation = sample["prompt"]
    expected_answer = sample["answer"]
    
    # Convert conversation to text
    prompt = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = generated_text[len(prompt):]
    
    # Compute reward
    reward = grade_output(completion, expected_answer)
    correct_count += reward
    
    # Extract question from conversation
    question = conversation[1]["content"]  # User message
    
    print(f"--- Sample {i+1} ---")
    print(f"Question: {question[:100]}...")
    print(f"Generated: {completion[:150]}...")
    print(f"Expected: {expected_answer}")
    print(f"Reward: {reward} {'✓' if reward == 1.0 else '✗'}\n")

accuracy = correct_count / len(test_samples)
print(f"Accuracy: {accuracy:.1%} ({int(correct_count)}/{len(test_samples)})")

# Log to wandb (optional)
try:
    wandb.log({
        "eval/accuracy": accuracy,
        "eval/correct_count": correct_count,
        "eval/total": len(test_samples),
    })
except:
    print("(Wandb logging skipped - not initialized)")

print("\n" + "="*80)
print("EXPERIMENTS COMPLETE")
print("="*80)

# %% [markdown]
# ## Next Steps
# 
# After running all cells successfully:
# 1. Adjust hyperparameters in the main training script
# 2. Run full training: `uv run python finetuning/train_grpo.py`
# 3. Monitor training in wandb
# 4. Experiment with different system prompts
# 5. Try different models or datasets
