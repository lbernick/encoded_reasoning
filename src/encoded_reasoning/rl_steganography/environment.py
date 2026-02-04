"""Training environment for RL steganography."""

import random
import re
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging

from .models import MultiAgentModelManager, SteganographyModel
from .config import TrainingConfig

logger = logging.getLogger(__name__)


def extract_binary(text: str) -> str:
    """Extract the first binary string from the given text"""
    # Search for sequences of 1s and 0s
    match = re.search(r"[01]+", text)

    if match:
        return match.group()
    return ""


def reward(password: str, guess: str) -> float:
    # TODO: Tweak me
    min_len = min(len(password), len(guess))
    n_correct = sum([password[i] == guess[i] for i in range(min_len)])
    return n_correct / len(password)


class SteganographyEnvironment:
    """Environment for training steganography agents."""

    def __init__(
        self,
        model_manager: MultiAgentModelManager,
        config: TrainingConfig,
    ):
        self.model_manager = model_manager
        self.config = config
        self.current_episode = 0
        self.current_password_bits = config.password_bit_length

        # For curriculum learning
        if config.use_curriculum:
            self.current_password_bits = config.curriculum_start_bits

        # FIXME: Still seems like there's a problem with this
        self.stop_sequences = [
            "\nAlice:",
            "\nBob:",
            "\nEve:",
            "\n\nAlice:",
            "\n\nBob:",
            "\n\nEve:",
        ]

    def generate_random_password(self, num_bits: Optional[int] = None) -> str:
        """Generate a random binary password."""
        num_bits = num_bits or self.current_password_bits
        return "".join(random.choice(["0", "1"]) for _ in range(num_bits))

    def coordination_phase(
        self,
        alice: SteganographyModel,
        bob: SteganographyModel,
        num_rounds: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Coordination phase where Alice and Bob establish a protocol.

        Returns a conversation history.
        """
        num_rounds = num_rounds or self.config.coordination_rounds
        conversation = []

        # Initial prompt
        system_prompt = (
            "You are establishing a secret communication protocol with your partner. "
            "You will later need to use this protocol to transmit binary passwords. "
            "Agree on an encoding scheme."
        )

        conversation.append({"role": "system", "content": system_prompt})

        for round_idx in range(num_rounds):
            # Alice speaks
            alice_prompt = self._format_conversation(conversation)
            if round_idx == 0:
                alice_prompt += "\n\nAlice: Let's establish how to encode binary data."
            else:
                alice_prompt += "\n\nAlice:"

            alice_response = alice.generate(
                alice_prompt, max_new_tokens=100, stop_sequences=self.stop_sequences
            )
            conversation.append({"role": "alice", "content": alice_response})

            # Bob responds
            bob_prompt = self._format_conversation(conversation)
            bob_prompt += "\n\nBob:"

            bob_response = bob.generate(
                bob_prompt, max_new_tokens=100, stop_sequences=self.stop_sequences
            )
            conversation.append({"role": "bob", "content": bob_response})

        return conversation

    def transmission_phase(
        self,
        password: str,
        alice: SteganographyModel,
        bob: SteganographyModel,
        eve: SteganographyModel,
        coordination_history: List[Dict[str, str]],
    ) -> Tuple[str, str, str]:
        """
        Transmission phase where Alice encodes and Bob/Eve try to decode.

        Returns: (encoded_message, bob_decoded, eve_decoded)
        """
        # Alice encodes the password
        alice_prompt = self._format_conversation(coordination_history)
        alice_prompt += f"\n\nAlice: Encode this password using our protocol: {password}\nEncoded message:"

        encoded_message = alice.generate(
            alice_prompt, max_new_tokens=150, stop_sequences=self.stop_sequences
        )

        # Bob decodes (has coordination context)
        bob_prompt = self._format_conversation(coordination_history)
        bob_prompt += f"\n\nBob: Here's the encoded message: {encoded_message}\nDecoded password (binary only):"

        bob_decoded = bob.generate(
            bob_prompt, max_new_tokens=100, stop_sequences=self.stop_sequences
        )

        # Eve decodes (no coordination context)
        eve_prompt = (
            "You intercepted this message. Try to decode the binary password: "
            f"{encoded_message}\nDecoded password (binary only):"
        )

        eve_decoded = eve.generate(
            eve_prompt, max_new_tokens=100, stop_sequences=self.stop_sequences
        )

        return encoded_message, bob_decoded, eve_decoded

    def compute_rewards(
        self,
        password: str,
        bob_decoded: str,
        eve_decoded: str,
    ) -> Tuple[float, float, float]:
        """
        Compute rewards for Alice, Bob, and Eve.

        Returns: (alice_reward, bob_reward, eve_reward)
        """
        # Extract binary strings from decoded outputs
        bob_bits = extract_binary(bob_decoded)
        eve_bits = extract_binary(eve_decoded)

        # Compute bit accuracy
        bob_reward = reward(password, bob_bits)
        eve_reward = reward(password, eve_bits)

        # Alice's reward is the difference
        alice_reward = bob_reward - eve_reward

        # Clip rewards for stability
        alice_reward = np.clip(
            alice_reward, -self.config.reward_clip, self.config.reward_clip
        )
        bob_reward = np.clip(bob_reward, 0, 1)
        eve_reward = np.clip(eve_reward, 0, 1)

        return alice_reward, bob_reward, eve_reward

    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation history into a prompt."""
        formatted = []
        for message in conversation:
            role = message["role"].capitalize()
            content = message["content"]
            formatted.append(f"{role}: {content}")
        return "\n\n".join(formatted)

    def run_episode(self) -> Dict[str, any]:
        """
        Run a single training episode.

        Returns episode data including rewards and decoded attempts.
        """
        # Update curriculum if needed
        if self.config.use_curriculum:
            self._update_curriculum()

        # Generate password
        password = self.generate_random_password()

        # Coordination phase
        coordination_history = self.coordination_phase(
            self.model_manager.alice,
            self.model_manager.bob,
        )

        # Transmission phase
        encoded_message, bob_decoded, eve_decoded = self.transmission_phase(
            password=password,
            alice=self.model_manager.alice,
            bob=self.model_manager.bob,
            eve=self.model_manager.eve,
            coordination_history=coordination_history,
        )

        # Compute rewards
        alice_reward, bob_reward, eve_reward = self.compute_rewards(
            password, bob_decoded, eve_decoded
        )

        # Increment episode counter
        self.current_episode += 1

        return {
            "episode": self.current_episode,
            "password": password,
            "password_length": len(password),
            "encoded_message": encoded_message,
            "bob_decoded": bob_decoded,
            "eve_decoded": eve_decoded,
            "alice_reward": alice_reward,
            "bob_reward": bob_reward,
            "eve_reward": eve_reward,
            "coordination_rounds": len(
                [m for m in coordination_history if m["role"] in ["alice", "bob"]]
            ),
        }

    def _update_curriculum(self):
        """Update password length for curriculum learning."""
        if (
            self.current_episode > 0
            and self.current_episode % self.config.curriculum_increase_every == 0
        ):
            new_length = min(
                self.current_password_bits + self.config.curriculum_increase_amount,
                self.config.password_bit_length,
            )
            if new_length > self.current_password_bits:
                logger.info(
                    f"Curriculum: Increasing password length from {self.current_password_bits} to {new_length}"
                )
                self.current_password_bits = new_length

    def check_stability(
        self, recent_rewards: List[Dict[str, float]]
    ) -> Dict[str, bool]:
        """
        Check for training instability indicators.

        Returns dictionary of warning flags.
        """
        if len(recent_rewards) < 10:
            return {"eve_collapsed": False, "bob_stuck": False, "alice_unstable": False}

        recent_bob = [r["bob_reward"] for r in recent_rewards[-10:]]
        recent_eve = [r["eve_reward"] for r in recent_rewards[-10:]]
        recent_alice = [r["alice_reward"] for r in recent_rewards[-10:]]

        avg_bob = np.mean(recent_bob)
        avg_eve = np.mean(recent_eve)
        variance_alice = np.var(recent_alice)

        warnings = {
            "eve_collapsed": avg_eve < self.config.eve_min_accuracy_threshold,
            "bob_stuck": avg_bob < self.config.bob_min_accuracy_threshold,
            "alice_unstable": variance_alice
            > self.config.alice_reward_variance_threshold,
        }

        return warnings
