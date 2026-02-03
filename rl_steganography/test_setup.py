"""Quick test script to verify the setup works."""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")

    try:
        from rl_steganography.config import (
            ModelConfig,
            TrainingConfig,
            EvaluationConfig,
        )

        logger.info("✓ Config modules imported")

        from rl_steganography.models import MultiAgentModelManager

        logger.info("✓ Models module imported")

        from rl_steganography.environment import SteganographyEnvironment

        logger.info("✓ Environment module imported")

        from rl_steganography.trainer import GRPOTrainer

        logger.info("✓ Trainer module imported")

        from rl_steganography.evaluator import SteganographyEvaluator

        logger.info("✓ Evaluator module imported")

        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_configs():
    """Test that configurations can be created."""
    logger.info("\nTesting configurations...")

    try:
        from rl_steganography.config import (
            ModelConfig,
            TrainingConfig,
            EvaluationConfig,
        )

        model_config = ModelConfig()
        logger.info(f"✓ ModelConfig created: {model_config.model_name}")

        training_config = TrainingConfig()
        logger.info(
            f"✓ TrainingConfig created: {training_config.num_episodes} episodes"
        )

        eval_config = EvaluationConfig()
        logger.info(
            f"✓ EvaluationConfig created: {eval_config.num_eval_episodes} eval episodes"
        )

        return True
    except Exception as e:
        logger.error(f"✗ Config creation failed: {e}")
        return False


def test_environment():
    """Test environment creation (without models)."""
    logger.info("\nTesting environment...")

    try:
        from rl_steganography.environment import SteganographyEnvironment
        from rl_steganography.config import TrainingConfig

        # Create config
        config = TrainingConfig(num_episodes=10)

        # Test password generation without full environment
        import random

        password = "".join(
            random.choice(["0", "1"]) for _ in range(config.password_bit_length)
        )
        logger.info(f"✓ Generated test password: {password[:16]}...")

        return True
    except Exception as e:
        logger.error(f"✗ Environment test failed: {e}")
        return False


def test_dependencies():
    """Test that key dependencies are installed."""
    logger.info("\nTesting dependencies...")

    dependencies = [
        "torch",
        "transformers",
        "peft",
        "trl",
        "accelerate",
        "bitsandbytes",
    ]

    all_present = True
    for dep in dependencies:
        try:
            __import__(dep)
            logger.info(f"✓ {dep} installed")
        except ImportError:
            logger.error(f"✗ {dep} NOT installed - run: pip install {dep}")
            all_present = False

    return all_present


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("RL Steganography - Quick Test")
    logger.info("=" * 80)

    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Configurations", test_configs),
        ("Environment", test_environment),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{name:<20} {status}")

    all_passed = all(result for _, result in results)

    if all_passed:
        logger.info("\n✓ All tests passed! Ready to run training.")
        logger.info("\nNext steps:")
        logger.info("  1. Make sure you have a CUDA GPU available")
        logger.info("  2. Run: python -m rl_steganography.train")
        return 0
    else:
        logger.info("\n✗ Some tests failed. Please install missing dependencies:")
        logger.info("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
