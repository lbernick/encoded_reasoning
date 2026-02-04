"""Training loop with GRPO for multi-agent steganography."""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import wandb
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import json
from datetime import datetime

from .models import MultiAgentModelManager
from .environment import SteganographyEnvironment
from .config import TrainingConfig

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) trainer for multi-agent steganography.

    Simplified version that focuses on policy gradient with relative advantage.
    """

    def __init__(
        self,
        model_manager: MultiAgentModelManager,
        environment: SteganographyEnvironment,
        config: TrainingConfig,
    ):
        self.model_manager = model_manager
        self.environment = environment
        self.config = config
        self.alice_bob_shared = model_manager.alice_bob_shared

        # Setup optimizers for each agent
        self.alice_optimizer = AdamW(
            self.model_manager.alice.base_model.parameters(),
            lr=config.learning_rate * config.alice_lr_multiplier,
        )

        # Bob optimizer only needed if Alice and Bob are separate
        if not self.alice_bob_shared:
            self.bob_optimizer = AdamW(
                self.model_manager.bob.base_model.parameters(),
                lr=config.learning_rate * config.bob_lr_multiplier,
            )
        else:
            self.bob_optimizer = None  # Bob uses Alice's optimizer

        self.eve_optimizer = AdamW(
            self.model_manager.eve.base_model.parameters(),
            lr=config.learning_rate * config.eve_lr_multiplier,
        )

        # Tracking
        self.episode_history = []
        self.training_start_time = datetime.now()

        # Setup logging
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        self.setup_logging()

        # Initialize Weights & Biases
        self.setup_wandb()

    def setup_logging(self):
        """Configure logging."""
        log_file = (
            Path(self.config.log_dir)
            / f"training_{self.training_start_time.strftime('%Y%m%d_%H%M%S')}.log"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    def setup_wandb(self):
        """Initialize Weights & Biases."""
        # Get model info for wandb config
        model_info = self.model_manager.get_model_info()

        # Initialize wandb
        wandb.init(
            project=self.config.wandb_project,
            name=f"run_{self.training_start_time.strftime('%Y%m%d_%H%M%S')}",
            config={
                # Model config
                "model_name": model_info["model_name"],
                "alice_bob_shared": model_info["alice_bob_shared"],
                "num_adapters": model_info["num_adapters"],
                "quantization": model_info["quantization"],
                "trainable_parameters": model_info["trainable_parameters"],
                # Training config
                "num_episodes": self.config.num_episodes,
                "learning_rate": self.config.learning_rate,
                "password_bit_length": self.config.password_bit_length,
                "coordination_rounds": self.config.coordination_rounds,
                "use_curriculum": self.config.use_curriculum,
                "curriculum_start_bits": self.config.curriculum_start_bits,
                "gradient_clip": self.config.gradient_clip,
                "reward_clip": self.config.reward_clip,
                # Learning rate multipliers
                "alice_lr_multiplier": self.config.alice_lr_multiplier,
                "bob_lr_multiplier": self.config.bob_lr_multiplier,
                "eve_lr_multiplier": self.config.eve_lr_multiplier,
            },
            dir=self.config.log_dir,
        )

        logger.info(f"Weights & Biases initialized: {wandb.run.url}")

    def compute_policy_loss(
        self,
        model,
        prompt: str,
        generated_text: str,
        reward: float,
        baseline_reward: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute policy gradient loss for a single agent.

        Uses advantage = reward - baseline for variance reduction.
        """
        # Tokenize the full sequence (prompt + generation)
        full_text = prompt + generated_text
        inputs = self.model_manager.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.base_model.device)

        # Get model outputs
        outputs = model.forward(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )

        logits = outputs.logits

        # Compute log probabilities of generated tokens
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Get log probs of actual tokens
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Compute advantage
        advantage = reward - baseline_reward

        # Policy gradient loss: -advantage * log_prob
        # Only compute over generated tokens (not prompt)
        prompt_length = len(self.model_manager.tokenizer.encode(prompt))
        generated_token_log_probs = token_log_probs[:, prompt_length - 1 :]

        # Mean over sequence length
        mean_log_prob = generated_token_log_probs.mean()

        # Policy loss
        loss = -advantage * mean_log_prob

        return loss

    def train_episode(self, episode_data: Dict) -> Dict[str, float]:
        """
        Train on a single episode using policy gradients.

        Returns training metrics.
        """
        # Compute baseline rewards (running average)
        baseline_alice = (
            np.mean([e["alice_reward"] for e in self.episode_history[-100:]])
            if len(self.episode_history) > 0
            else 0.0
        )
        baseline_bob = (
            np.mean([e["bob_reward"] for e in self.episode_history[-100:]])
            if len(self.episode_history) > 0
            else 0.5
        )
        baseline_eve = (
            np.mean([e["eve_reward"] for e in self.episode_history[-100:]])
            if len(self.episode_history) > 0
            else 0.5
        )

        # Reconstruct prompts and generations
        # For simplicity, we'll use a simplified version
        # In a full implementation, we'd store the exact prompts used

        password = episode_data["password"]
        encoded_message = episode_data["encoded_message"]
        bob_decoded = episode_data["bob_decoded"]
        eve_decoded = episode_data["eve_decoded"]

        alice_reward = episode_data["alice_reward"]
        bob_reward = episode_data["bob_reward"]
        eve_reward = episode_data["eve_reward"]

        # Simplified prompts (in real implementation, store from environment)
        alice_prompt = f"Encode this password: {password}\nEncoded:"
        bob_prompt = f"Decode this message: {encoded_message}\nDecoded:"
        eve_prompt = f"Decode this intercepted message: {encoded_message}\nDecoded:"

        losses = {}

        if self.alice_bob_shared:
            # Alice and Bob share the same adapter, so we train once with combined reward
            # Combined reward for Alice/Bob: focus on Bob's decoding accuracy
            combined_reward = bob_reward + alice_reward  # Bob's accuracy + gap vs Eve

            self.alice_optimizer.zero_grad()

            # Train on Alice's encoding task
            alice_loss = self.compute_policy_loss(
                self.model_manager.alice,
                alice_prompt,
                encoded_message,
                combined_reward,
                baseline_alice + baseline_bob,
            )

            # Train on Bob's decoding task (same adapter)
            bob_loss = self.compute_policy_loss(
                self.model_manager.bob,
                bob_prompt,
                bob_decoded,
                bob_reward,
                baseline_bob,
            )

            # Combine losses
            total_loss = alice_loss + bob_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model_manager.alice.base_model.parameters(),
                self.config.gradient_clip,
            )
            self.alice_optimizer.step()

            losses["alice_bob_loss"] = total_loss.item()
            losses["alice_loss"] = alice_loss.item()
            losses["bob_loss"] = bob_loss.item()
        else:
            # Alice and Bob have separate adapters, train independently

            # Train Alice
            self.alice_optimizer.zero_grad()
            alice_loss = self.compute_policy_loss(
                self.model_manager.alice,
                alice_prompt,
                encoded_message,
                alice_reward,
                baseline_alice,
            )
            alice_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model_manager.alice.base_model.parameters(),
                self.config.gradient_clip,
            )
            self.alice_optimizer.step()
            losses["alice_loss"] = alice_loss.item()

            # Train Bob
            self.bob_optimizer.zero_grad()
            bob_loss = self.compute_policy_loss(
                self.model_manager.bob,
                bob_prompt,
                bob_decoded,
                bob_reward,
                baseline_bob,
            )
            bob_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model_manager.bob.base_model.parameters(),
                self.config.gradient_clip,
            )
            self.bob_optimizer.step()
            losses["bob_loss"] = bob_loss.item()

        # Train Eve (always separate)
        self.eve_optimizer.zero_grad()
        eve_loss = self.compute_policy_loss(
            self.model_manager.eve,
            eve_prompt,
            eve_decoded,
            eve_reward,
            baseline_eve,
        )
        eve_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model_manager.eve.base_model.parameters(), self.config.gradient_clip
        )
        self.eve_optimizer.step()
        losses["eve_loss"] = eve_loss.item()

        return losses

    def train(self):
        """Main training loop."""
        logger.info("Starting training")
        logger.info(f"Configuration: {self.config}")

        # Log model info
        model_info = self.model_manager.get_model_info()
        logger.info(f"Model info: {model_info}")

        progress_bar = tqdm(range(self.config.num_episodes), desc="Training")

        for episode_idx in progress_bar:
            # Run episode
            episode_data = self.environment.run_episode()
            self.episode_history.append(episode_data)

            # Train on episode
            losses = self.train_episode(episode_data)

            # Log metrics
            if episode_idx % self.config.log_every == 0:
                self._log_metrics(episode_idx, episode_data, losses)

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "alice_r": f"{episode_data['alice_reward']:.3f}",
                        "bob_r": f"{episode_data['bob_reward']:.3f}",
                        "eve_r": f"{episode_data['eve_reward']:.3f}",
                    }
                )

            # Log example conversations to W&B
            if episode_idx % self.config.log_examples_every == 0 and episode_idx > 0:
                self._log_example_to_wandb(episode_data)

            # Check for stability issues
            if episode_idx % self.config.eval_every == 0 and episode_idx > 0:
                warnings = self.environment.check_stability(self.episode_history)
                if any(warnings.values()):
                    logger.warning(
                        f"Stability warnings at episode {episode_idx}: {warnings}"
                    )
                    self._handle_stability_issues(warnings)

            # Checkpoint
            if episode_idx % self.config.checkpoint_every == 0 and episode_idx > 0:
                self._save_checkpoint(episode_idx)

        logger.info("Training complete")
        self._save_final_results()

        # Finish wandb run
        wandb.finish()

    def _log_metrics(self, episode_idx: int, episode_data: Dict, losses: Dict):
        """Log training metrics to Weights & Biases."""
        # Compute running averages
        avg_alice = avg_bob = avg_eve = None
        if len(self.episode_history) >= 10:
            recent = self.episode_history[-10:]
            avg_alice = np.mean([e["alice_reward"] for e in recent])
            avg_bob = np.mean([e["bob_reward"] for e in recent])
            avg_eve = np.mean([e["eve_reward"] for e in recent])

        # Log to W&B
        log_dict = {
            # Rewards
            "reward/alice": episode_data["alice_reward"],
            "reward/bob": episode_data["bob_reward"],
            "reward/eve": episode_data["eve_reward"],
            "reward/gap": episode_data["alice_reward"],  # Alice's reward is Bob-Eve gap
            # Losses
            "loss/alice": losses["alice_loss"],
            "loss/bob": losses["bob_loss"],
            "loss/eve": losses["eve_loss"],
            # Environment
            "environment/password_length": episode_data["password_length"],
            "environment/coordination_rounds": episode_data["coordination_rounds"],
            # Episode
            "episode": episode_idx,
        }

        # Add running averages if available
        if avg_alice is not None:
            log_dict.update(
                {
                    "reward_avg/alice": avg_alice,
                    "reward_avg/bob": avg_bob,
                    "reward_avg/eve": avg_eve,
                    "reward_avg/gap": avg_bob - avg_eve,
                }
            )

        # Add combined loss if in shared mode
        if "alice_bob_loss" in losses:
            log_dict["loss/alice_bob_combined"] = losses["alice_bob_loss"]

        wandb.log(log_dict, step=episode_idx)

    def _log_example_to_wandb(self, episode_data: Dict):
        """Log an example conversation to W&B as a table."""
        # Create a W&B table with the example
        table = wandb.Table(
            columns=[
                "Password",
                "Encoded Message",
                "Bob Decoded",
                "Eve Decoded",
                "Bob Accuracy",
                "Eve Accuracy",
                "Gap",
            ]
        )

        table.add_data(
            episode_data["password"],
            episode_data["encoded_message"],
            episode_data["bob_decoded"],
            episode_data["eve_decoded"],
            f"{episode_data['bob_reward']:.2%}",
            f"{episode_data['eve_reward']:.2%}",
            f"{episode_data['alice_reward']:.3f}",
        )

        wandb.log({"examples": table, "episode": episode_data["episode"]})

    def _handle_stability_issues(self, warnings: Dict[str, bool]):
        """Handle training stability issues."""
        if warnings["eve_collapsed"]:
            logger.warning(
                "Eve has collapsed! Consider resetting Eve or adjusting learning rates."
            )
            # Could implement automatic reset here

        if warnings["bob_stuck"]:
            logger.warning("Bob is not learning effectively.")

        if warnings["alice_unstable"]:
            logger.warning("Alice's rewards are highly unstable.")

    def _save_checkpoint(self, episode_idx: int):
        """Save model checkpoint."""
        logger.info(f"Saving checkpoint at episode {episode_idx}")
        self.model_manager.save_all_adapters(self.config.output_dir, episode_idx)

        # Save training state
        checkpoint_path = (
            Path(self.config.output_dir)
            / f"episode_{episode_idx}"
            / "training_state.json"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, "w") as f:
            json.dump(
                {
                    "episode": episode_idx,
                    "config": self.config.__dict__,
                    "recent_rewards": self.episode_history[-100:]
                    if len(self.episode_history) > 100
                    else self.episode_history,
                },
                f,
                indent=2,
            )

    def _save_final_results(self):
        """Save final training results."""
        results_path = Path(self.config.output_dir) / "final_results.json"

        with open(results_path, "w") as f:
            json.dump(
                {
                    "config": self.config.__dict__,
                    "total_episodes": len(self.episode_history),
                    "final_rewards": {
                        "alice": self.episode_history[-1]["alice_reward"],
                        "bob": self.episode_history[-1]["bob_reward"],
                        "eve": self.episode_history[-1]["eve_reward"],
                    },
                    "average_rewards_last_100": {
                        "alice": np.mean(
                            [e["alice_reward"] for e in self.episode_history[-100:]]
                        ),
                        "bob": np.mean(
                            [e["bob_reward"] for e in self.episode_history[-100:]]
                        ),
                        "eve": np.mean(
                            [e["eve_reward"] for e in self.episode_history[-100:]]
                        ),
                    },
                },
                f,
                indent=2,
            )

        logger.info(f"Final results saved to {results_path}")
