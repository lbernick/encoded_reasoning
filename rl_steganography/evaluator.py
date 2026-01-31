"""Evaluation framework for steganography models."""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime
from collections import defaultdict

from .models import MultiAgentModelManager
from .environment import SteganographyEnvironment
from .config import EvaluationConfig

logger = logging.getLogger(__name__)


class SteganographyEvaluator:
    """Evaluator for trained steganography models."""
    
    def __init__(
        self,
        model_manager: MultiAgentModelManager,
        environment: SteganographyEnvironment,
        config: EvaluationConfig,
    ):
        self.model_manager = model_manager
        self.environment = environment
        self.config = config
    
    def evaluate(
        self,
        num_episodes: Optional[int] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Run evaluation and return comprehensive metrics.
        
        Args:
            num_episodes: Number of episodes to evaluate (default from config)
            save_dir: Directory to save evaluation results
        
        Returns:
            Dictionary of evaluation metrics
        """
        num_episodes = num_episodes or self.config.num_eval_episodes
        save_dir = save_dir or f"./eval_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting evaluation with {num_episodes} episodes")
        
        # Store all episode results
        all_episodes = []
        examples_to_save = []
        
        # Run evaluation episodes
        for ep_idx in range(num_episodes):
            episode_data = self.environment.run_episode()
            all_episodes.append(episode_data)
            
            # Save interesting examples
            if self.config.save_examples and len(examples_to_save) < self.config.max_examples_to_save:
                examples_to_save.append({
                    "episode": ep_idx,
                    "password": episode_data["password"],
                    "encoded_message": episode_data["encoded_message"],
                    "bob_decoded": episode_data["bob_decoded"],
                    "eve_decoded": episode_data["eve_decoded"],
                    "bob_reward": episode_data["bob_reward"],
                    "eve_reward": episode_data["eve_reward"],
                })
        
        # Compute metrics
        metrics = self._compute_metrics(all_episodes)
        
        # Add length-specific metrics
        length_metrics = self._compute_length_specific_metrics(all_episodes)
        metrics["by_length"] = length_metrics
        
        # Analyze encoding schemes
        encoding_analysis = self._analyze_encoding_schemes(all_episodes)
        metrics["encoding_analysis"] = encoding_analysis
        
        # Save results
        if save_dir:
            self._save_results(metrics, examples_to_save, save_dir)
        
        return metrics
    
    def _compute_metrics(self, episodes: List[Dict]) -> Dict:
        """Compute aggregate metrics across episodes."""
        alice_rewards = [e["alice_reward"] for e in episodes]
        bob_rewards = [e["bob_reward"] for e in episodes]
        eve_rewards = [e["eve_reward"] for e in episodes]
        
        metrics = {
            "num_episodes": len(episodes),
            "alice": {
                "mean": np.mean(alice_rewards),
                "std": np.std(alice_rewards),
                "min": np.min(alice_rewards),
                "max": np.max(alice_rewards),
            },
            "bob": {
                "mean": np.mean(bob_rewards),
                "std": np.std(bob_rewards),
                "min": np.min(bob_rewards),
                "max": np.max(bob_rewards),
                "accuracy": np.mean(bob_rewards),  # Reward is accuracy
            },
            "eve": {
                "mean": np.mean(eve_rewards),
                "std": np.std(eve_rewards),
                "min": np.min(eve_rewards),
                "max": np.max(eve_rewards),
                "accuracy": np.mean(eve_rewards),
            },
            "gap": {
                "bob_minus_eve": np.mean([b - e for b, e in zip(bob_rewards, eve_rewards)]),
                "success_rate": np.mean([1 if b > e else 0 for b, e in zip(bob_rewards, eve_rewards)]),
            },
        }
        
        return metrics
    
    def _compute_length_specific_metrics(self, episodes: List[Dict]) -> Dict:
        """Compute metrics broken down by password length."""
        by_length = defaultdict(list)
        
        for episode in episodes:
            length = episode["password_length"]
            by_length[length].append({
                "bob_reward": episode["bob_reward"],
                "eve_reward": episode["eve_reward"],
            })
        
        length_metrics = {}
        for length, data in by_length.items():
            bob_rewards = [d["bob_reward"] for d in data]
            eve_rewards = [d["eve_reward"] for d in data]
            
            length_metrics[length] = {
                "count": len(data),
                "bob_accuracy": np.mean(bob_rewards),
                "eve_accuracy": np.mean(eve_rewards),
                "gap": np.mean([b - e for b, e in zip(bob_rewards, eve_rewards)]),
            }
        
        return length_metrics
    
    def _analyze_encoding_schemes(self, episodes: List[Dict]) -> Dict:
        """Analyze the encoding schemes that emerged."""
        analysis = {
            "message_length_stats": {},
            "character_diversity": {},
            "pattern_analysis": {},
        }
        
        # Message length statistics
        encoded_lengths = [len(e["encoded_message"]) for e in episodes]
        password_lengths = [len(e["password"]) for e in episodes]
        
        analysis["message_length_stats"] = {
            "mean_encoded_length": np.mean(encoded_lengths),
            "std_encoded_length": np.std(encoded_lengths),
            "mean_password_length": np.mean(password_lengths),
            "compression_ratio": np.mean([el / pl if pl > 0 else 0 for el, pl in zip(encoded_lengths, password_lengths)]),
        }
        
        # Character diversity in encoded messages
        all_chars = set()
        char_counts = defaultdict(int)
        
        for episode in episodes:
            for char in episode["encoded_message"]:
                all_chars.add(char)
                char_counts[char] += 1
        
        analysis["character_diversity"] = {
            "unique_chars": len(all_chars),
            "total_chars": sum(char_counts.values()),
            "most_common": sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        }
        
        # Look for patterns
        # Simple analysis: Are encoded messages using binary directly?
        binary_chars_ratio = []
        for episode in episodes:
            msg = episode["encoded_message"]
            binary_chars = sum(1 for c in msg if c in ['0', '1'])
            binary_chars_ratio.append(binary_chars / len(msg) if len(msg) > 0 else 0)
        
        analysis["pattern_analysis"] = {
            "binary_char_ratio": {
                "mean": np.mean(binary_chars_ratio),
                "std": np.std(binary_chars_ratio),
            },
            "uses_mostly_binary": np.mean(binary_chars_ratio) > 0.8,
        }
        
        return analysis
    
    def _save_results(self, metrics: Dict, examples: List[Dict], save_dir: str):
        """Save evaluation results to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = save_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Save examples
        if examples:
            examples_file = save_path / "examples.json"
            with open(examples_file, 'w') as f:
                json.dump(examples, f, indent=2)
            logger.info(f"Saved {len(examples)} examples to {examples_file}")
        
        # Save human-readable summary
        summary_file = save_path / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STEGANOGRAPHY EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Number of episodes: {metrics['num_episodes']}\n\n")
            
            f.write("Overall Performance:\n")
            f.write(f"  Bob Accuracy:  {metrics['bob']['accuracy']:.3f} ± {metrics['bob']['std']:.3f}\n")
            f.write(f"  Eve Accuracy:  {metrics['eve']['accuracy']:.3f} ± {metrics['eve']['std']:.3f}\n")
            f.write(f"  Gap (Bob-Eve): {metrics['gap']['bob_minus_eve']:.3f}\n")
            f.write(f"  Bob Wins:      {metrics['gap']['success_rate']:.1%}\n\n")
            
            f.write("Performance by Password Length:\n")
            for length, stats in sorted(metrics['by_length'].items()):
                f.write(f"  {length} bits: Bob={stats['bob_accuracy']:.3f}, Eve={stats['eve_accuracy']:.3f}, Gap={stats['gap']:.3f}\n")
            
            f.write("\nEncoding Analysis:\n")
            enc = metrics['encoding_analysis']
            f.write(f"  Avg encoded length: {enc['message_length_stats']['mean_encoded_length']:.1f} chars\n")
            f.write(f"  Compression ratio:  {enc['message_length_stats']['compression_ratio']:.2f}x\n")
            f.write(f"  Unique characters:  {enc['character_diversity']['unique_chars']}\n")
            f.write(f"  Binary char ratio:  {enc['pattern_analysis']['binary_char_ratio']['mean']:.2%}\n")
            f.write(f"  Uses mostly binary: {enc['pattern_analysis']['uses_mostly_binary']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Saved summary to {summary_file}")
    
    def evaluate_generalization(self) -> Dict:
        """
        Test generalization to different password lengths.
        
        Returns metrics for each tested length.
        """
        logger.info("Evaluating generalization across password lengths")
        
        results = {}
        original_length = self.environment.current_password_bits
        
        for length in self.config.eval_password_lengths:
            logger.info(f"Testing with {length}-bit passwords")
            self.environment.current_password_bits = length
            
            episodes = []
            for _ in range(20):  # Run 20 episodes per length
                episode_data = self.environment.run_episode()
                episodes.append(episode_data)
            
            metrics = self._compute_metrics(episodes)
            results[length] = metrics
        
        # Restore original length
        self.environment.current_password_bits = original_length
        
        return results
    
    def human_interpretability_test(self, num_samples: int = 10) -> List[Dict]:
        """
        Generate samples for human interpretability testing.
        
        Returns list of examples with full context for human analysis.
        """
        logger.info(f"Generating {num_samples} samples for human interpretability test")
        
        samples = []
        for i in range(num_samples):
            episode_data = self.environment.run_episode()
            
            samples.append({
                "sample_id": i + 1,
                "password": episode_data["password"],
                "encoded_message": episode_data["encoded_message"],
                "bob_decoded": episode_data["bob_decoded"],
                "bob_correct": episode_data["bob_reward"],
                "question": "Can you identify the encoding pattern used?",
            })
        
        return samples
