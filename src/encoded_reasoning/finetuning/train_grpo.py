import argparse
import os
import subprocess
import sys
from collections import namedtuple
from dataclasses import dataclass
import random
from typing_extensions import Self
import torch
import wandb
from functools import partial
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple
from datasets import load_dataset as hf_load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from huggingface_hub import login
from .grader import constrained_reasoning_substring, parse_reasoning_and_answer
from ..evals.constraints import CONSTRAINTS, ReasoningConstraint
from ..evals.datasets import DATASETS, DatasetRecipe
from ..evals.prompts import get_base_system_prompt, get_example


@dataclass
class FinetuningArgs:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "gsm8k"
    constraint: str = "only_emojis"
    use_curriculum_learning: bool = False

    learning_rate: float = 5e-6
    batch_size: int = 4
    num_train_epochs: int = 1
    max_steps: int = -1  # -1 for unlimited
    gradient_accumulation_steps: int = 4

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: int = 0.05
    lora_target_modules: Tuple[str] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    max_new_tokens: int = 512
    temperature: float = 0.7

    # Logging
    log_every_n_steps: int = 10
    save_every_n_steps: int = 50
    wandb_project: str = None
    wandb_run_name: str = None
    output_dir: str = None


class HyperparamScheduler:
    def __init__(self, initial_value, **kwargs):
        self.initial_value = initial_value
        self.kwargs = kwargs
        self.step_count = 0

    def step(self):
        self.step_count += 1
        return self.get_value()

    def get_value(self):
        final_value = self.kwargs.get("final_value", 0)
        total_steps = self.kwargs["total_steps"]
        progress = min(self.step_count / total_steps, 1.0)
        return self.initial_value + (final_value - self.initial_value) * progress


class Trainer:
    def __init__(self, args: FinetuningArgs):
        model_name = args.model_name.split("/")[-1]
        if not args.wandb_project:
            args.wandb_project = f"{model_name}-{args.dataset_name}-grpo"
        if not args.wandb_run_name:
            args.wandb_run_name = (
                f"{model_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        if not args.output_dir:
            args.output_dir = f"./outputs/{args.wandb_project}"

        self.args = args
        self.dataset_recipe = DATASETS[args.dataset_name]
        self.constraint = CONSTRAINTS[args.constraint]
        # FIXME: Dedupe with code in runner.py
        dataset_type = self.dataset_recipe["type"]
        base_system_prompt = get_base_system_prompt(reasoning=True, dataset_type=dataset_type)
        example = get_example(self.constraint.reasoning_example, dataset_type)
        full_prompt = base_system_prompt + "\n" + self.constraint.system_prompt + "\n" + self.dataset_recipe.get("system_prompt", "")
        full_prompt += example
        self.system_prompt = full_prompt
        # self.reward_function = objective.reward_function
        self.scheduler = HyperparamScheduler(
            initial_value=1, final_value=0, total_steps=args.max_steps
        )

    def setup_model_and_tokenizer(self):
        """Initialize model with 4-bit quantization and LoRA adapters."""
        print(f"Loading model: {self.args.model_name}")

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Always use left padding for decoder-only models (AutoModelForCausalLM)
        tokenizer.padding_side = "left"

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules=self.args.lora_target_modules,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Add LoRA adapters
        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )

        return model, tokenizer

    def prepare_dataset(
        self, custom_dataset: DatasetRecipe, system_prompt: str, n_samples: int = None
    ):
        """Load and format the dataset for training."""
        print(f"Loading {self.args.dataset_name} dataset...")
        dataset = hf_load_dataset(
            custom_dataset["hf_path"], custom_dataset["config"], split=custom_dataset["train_split"]
        )

        if n_samples is not None:
            indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
            dataset = dataset.select(indices)

        print(f"Dataset loaded: {len(dataset)} examples")

        def format_example(example):
            """Format a single example."""
            question, answer = custom_dataset["format_func"](example)
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]

            return {
                "prompt": conversation,
                "answer": answer,
            }

        formatted_dataset = dataset.map(
            format_example, remove_columns=dataset.column_names
        )

        # Print a sample
        print("\n" + "=" * 80)
        print("SAMPLE FORMATTED EXAMPLE:")
        print("=" * 80)
        sample = formatted_dataset[0]
        print("Conversation:")
        for msg in sample["prompt"]:
            print(f"  [{msg['role']}]: {msg['content']}")
        print(f"\nExpected answer: {sample['answer']}")
        print("=" * 80 + "\n")

        return formatted_dataset

    def reward_function(self, generated_text, correct_answer, percent_reasoning_allowed) -> float:
        reasoning, generated_answer = parse_reasoning_and_answer(generated_text)
        if reasoning is None and generated_answer is None:
            return -1.0
        elif reasoning is None or generated_answer is None:
            return -0.75
        if generated_answer != correct_answer:
            return -0.5
        reasoning = constrained_reasoning_substring(reasoning, percent_reasoning_allowed)
        if len(reasoning) == 0:
            return 1.0
        reward = self.constraint.reward_function(reasoning)
        return reward

    
    def compute_rewards(self, completions: List[str], **kwargs) -> List[float]:
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
        prompts = kwargs["prompts"]

        for i, output in enumerate(completions):
            correct_answer = correct_answers[i] if i < len(correct_answers) else ""
            generated_answer = output[0]["content"]
            prompt = prompts[
                i
            ][
                1
            ]  # First message is the system prompt, second message is the dataset question
            if self.args.use_curriculum_learning:
                percent_reasoning_allowed = self.scheduler.get_value()
            else:
                percent_reasoning_allowed = 0

            try:
                # Get correct answer for this example
                reward = self.reward_function(
                    generated_answer, correct_answer, percent_reasoning_allowed
                )
                rewards.append(reward)

                # Log sample outputs periodically
                if i == 0:  # Log first sample in batch
                    print(f"\n{'=' * 80}")
                    print(f"PROMPT:\n{prompt}")
                    print(f"SAMPLE OUTPUT (Reward: {reward})")
                    print(f"{'=' * 80}")
                    print(f"Output:\n{generated_answer}")
                    print(f"Expected answer: {correct_answer}")
                    print(f"{'=' * 80}\n")
                    wandb.log(
                        {
                            "sample output": output,
                            "prompt": prompt,
                            "expected_answer": correct_answer,
                            "reward": reward,
                        }
                    )

            except Exception as e:
                print(f"Error computing reward for sample {i}: {e}")
                rewards.append(0.0)

        self.scheduler.step()
        if rewards:
            mean_reward = sum(rewards) / len(rewards)
            print(
                f"Batch rewards - Mean: {mean_reward:.3f}, Min: {min(rewards):.3f}, Max: {max(rewards):.3f}"
            )

            # Log to wandb
            wandb.log(
                {
                    "reward/mean": mean_reward,
                    "reward/std": torch.tensor(rewards).std().item()
                    if len(rewards) > 1
                    else 0.0,
                    "reward/min": min(rewards),
                    "reward/max": max(rewards),
                    "percent_reasoning_allowed_to_be_unconstrained": percent_reasoning_allowed,
                }
            )

        return rewards

    def train(self):
        print("=" * 80)
        print("RL TRAINING WITH GRPO")
        print("=" * 80)

        wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_run_name,
            config={
                "model": self.args.model_name,
                "dataset": self.args.dataset_name,
                "learning_rate": self.args.learning_rate,
                "batch_size": self.args.batch_size,
                "lora_r": self.args.lora_r,
                "lora_alpha": self.args.lora_alpha,
                "max_steps": self.args.max_steps,
                "system_prompt": self.system_prompt,
            },
        )

        train_dataset = self.prepare_dataset(self.dataset_recipe, self.system_prompt)
        model, tokenizer = self.setup_model_and_tokenizer()

        # https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig
        training_args = GRPOConfig(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            max_steps=self.args.max_steps,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            logging_steps=self.args.log_every_n_steps,
            save_steps=self.args.save_every_n_steps,
            save_total_limit=3,
            report_to="wandb",
            remove_unused_columns=False,
            temperature=self.args.temperature,
        )

        # Disable Qwen3's built-in thinking mode so model follows our system prompt
        # (otherwise it outputs <think> tags instead of <reasoning> tags)
        model.generation_config.enable_thinking = False

        # https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOTrainer
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            reward_funcs=self.compute_rewards,
        )

        print("\nStarting training...")
        try:
            trainer.train()
        finally:
            try:
                print(f"\nSaving model to {self.args.output_dir}")
                trainer.save_model(self.args.output_dir)
                trainer.push_to_hub(
                    commit_message=self.args.wandb_run_name,
                )
            finally:
                wandb.finish()

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)


def check_git_clean():
    """Check if the git repository has uncommitted changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    )

    if result.stdout.strip():
        print(
            "Error: You have uncommitted changes in your repository:", file=sys.stderr
        )
        print(result.stdout, file=sys.stderr)
        print("\nPlease commit or stash your changes before training.", file=sys.stderr)
        sys.exit(1)

    print("✓ Git repository is clean")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using GRPO")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gsm8k",
        help="Dataset name (default: gsm8k)",
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default="only_emojis",
        help="Constraint type (default: only_emojis)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum training steps, -1 for unlimited (default: -1)",
    )
    
    cli_args = parser.parse_args()
    
    check_git_clean()
    load_dotenv()
    login()
    os.environ["WANDB_GIT_COMMIT"] = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
    
    args = FinetuningArgs(
        model_name=cli_args.model_name,
        dataset_name=cli_args.dataset_name,
        constraint=cli_args.constraint,
        max_steps=cli_args.max_steps,
    )
    trainer = Trainer(args)
    trainer.train()
