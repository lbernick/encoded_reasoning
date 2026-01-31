"""Main training script for RL steganography."""

import argparse
import logging
from pathlib import Path

from rl_steganography.config import ModelConfig, TrainingConfig, EvaluationConfig
from rl_steganography.models import MultiAgentModelManager
from rl_steganography.environment import SteganographyEnvironment
from rl_steganography.trainer import GRPOTrainer
from rl_steganography.evaluator import SteganographyEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train steganography agents with RL")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model to use")
    parser.add_argument("--use-4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (Q-LoRA)")
    parser.add_argument("--no-4bit", action="store_false", dest="use_4bit",
                        help="Don't use 4-bit quantization")
    
    # Training arguments
    parser.add_argument("--num-episodes", type=int, default=10000,
                        help="Number of training episodes")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--password-bits", type=int, default=32,
                        help="Password length in bits")
    parser.add_argument("--coordination-rounds", type=int, default=3,
                        help="Number of coordination rounds")
    
    # Curriculum learning
    parser.add_argument("--use-curriculum", action="store_true", default=True,
                        help="Use curriculum learning")
    parser.add_argument("--no-curriculum", action="store_false", dest="use_curriculum",
                        help="Don't use curriculum learning")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./outputs/rl_steganography",
                        help="Output directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="./logs/rl_steganography",
                        help="Directory for logs")
    
    # Mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode: train or eval")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to load for evaluation")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("RL Steganography Training")
    logger.info("=" * 80)
    
    # Create configurations
    model_config = ModelConfig(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
    )
    
    training_config = TrainingConfig(
        num_episodes=args.num_episodes,
        learning_rate=args.learning_rate,
        password_bit_length=args.password_bits,
        coordination_rounds=args.coordination_rounds,
        use_curriculum=args.use_curriculum,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )
    
    eval_config = EvaluationConfig()
    
    # Initialize model manager
    logger.info("Initializing models...")
    model_manager = MultiAgentModelManager(model_config)
    model_manager.initialize()
    
    # Print model info
    model_info = model_manager.get_model_info()
    logger.info(f"Model: {model_info['model_name']}")
    logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,} ({model_info['trainable_percentage']:.2f}%)")
    
    # Create environment
    environment = SteganographyEnvironment(model_manager, training_config)
    
    if args.mode == "train":
        # Train
        trainer = GRPOTrainer(model_manager, environment, training_config)
        trainer.train()
        
        # Run final evaluation
        logger.info("Running final evaluation...")
        evaluator = SteganographyEvaluator(model_manager, environment, eval_config)
        metrics = evaluator.evaluate(save_dir=Path(args.output_dir) / "final_eval")
        
        logger.info("=" * 80)
        logger.info("FINAL RESULTS")
        logger.info("=" * 80)
        logger.info(f"Bob accuracy: {metrics['bob']['accuracy']:.3f} ± {metrics['bob']['std']:.3f}")
        logger.info(f"Eve accuracy: {metrics['eve']['accuracy']:.3f} ± {metrics['eve']['std']:.3f}")
        logger.info(f"Gap (Bob-Eve): {metrics['gap']['bob_minus_eve']:.3f}")
        logger.info(f"Bob wins: {metrics['gap']['success_rate']:.1%}")
        
    elif args.mode == "eval":
        # Evaluate existing checkpoint
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            # TODO: Implement checkpoint loading
            logger.warning("Checkpoint loading not yet implemented")
        
        evaluator = SteganographyEvaluator(model_manager, environment, eval_config)
        
        # Run comprehensive evaluation
        logger.info("Running evaluation...")
        metrics = evaluator.evaluate()
        
        logger.info("Testing generalization...")
        generalization_results = evaluator.evaluate_generalization()
        
        logger.info("Generating interpretability samples...")
        samples = evaluator.human_interpretability_test(num_samples=10)
        
        # Print results
        logger.info("=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Bob accuracy: {metrics['bob']['accuracy']:.3f} ± {metrics['bob']['std']:.3f}")
        logger.info(f"Eve accuracy: {metrics['eve']['accuracy']:.3f} ± {metrics['eve']['std']:.3f}")
        logger.info(f"Gap (Bob-Eve): {metrics['gap']['bob_minus_eve']:.3f}")
        
        logger.info("\nGeneralization:")
        for length, result in generalization_results.items():
            logger.info(f"  {length} bits: Bob={result['bob']['accuracy']:.3f}, Eve={result['eve']['accuracy']:.3f}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
