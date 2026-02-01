"""Integration tests for multi-agent model manager with shared adapters.

This test suite verifies the actual behavior of the multi-adapter system by loading
real models and testing generation. Unlike test_models.py which uses mocks for unit
tests, this file performs integration testing with actual model loading.

This test suite verifies that:
1. When shared_alice_bob=True, Alice and Bob share the same adapter and produce identical outputs
2. When shared_alice_bob=False, Alice and Bob have separate adapters
3. All agents properly use multi-adapter architecture with a shared base model

Usage:
    # Run all tests (shared mode only, by default)
    python -m unittest rl_steganography.test_shared_adapter -v
    
    # Run specific test class
    python -m unittest rl_steganography.test_shared_adapter.TestSharedAdapter -v
    
    # Run specific test method
    python -m unittest rl_steganography.test_shared_adapter.TestSharedAdapter.test_alice_bob_identical_responses_greedy -v
    
    # Run from the test file directly
    python rl_steganography/test_shared_adapter.py
    
    # To also test separate adapter mode (takes longer), edit the suite() function

Note: These are integration tests that require GPU and will take 2-5 minutes to load models.
      For faster unit tests with mocks, see test_models.py
"""

import unittest
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl_steganography.config import ModelConfig
from rl_steganography.models import MultiAgentModelManager


class TestSharedAdapter(unittest.TestCase):
    """Test that Alice and Bob share adapters correctly when shared_alice_bob=True."""
    
    @classmethod
    def setUpClass(cls):
        """Load models once for all tests."""
        cls.model_config = ModelConfig(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            shared_alice_bob=True,
            use_4bit=True,
            lora_r=64,
            max_new_tokens=128,
            temperature=0.7,
        )
        
        print(f"\nLoading models: {cls.model_config.model_name}")
        print(f"Configuration: {'SHARED' if cls.model_config.shared_alice_bob else 'SEPARATE'} adapters")
        print("This may take 2-5 minutes...\n")
        
        cls.model_manager = MultiAgentModelManager(cls.model_config)
        cls.model_manager.initialize()
        
        print("Models loaded successfully!\n")
    
    def test_alice_bob_same_object(self):
        """Test that Alice and Bob reference the same SteganographyModel object."""
        self.assertIs(
            self.model_manager.alice,
            self.model_manager.bob,
            "Alice and Bob should be the same object when shared_alice_bob=True"
        )
    
    def test_adapter_names(self):
        """Test that adapter names are set correctly."""
        self.assertEqual(
            self.model_manager.alice.adapter_name,
            "alice_bob",
            "Alice should use 'alice_bob' adapter"
        )
        self.assertEqual(
            self.model_manager.bob.adapter_name,
            "alice_bob",
            "Bob should use 'alice_bob' adapter"
        )
        self.assertEqual(
            self.model_manager.eve.adapter_name,
            "eve",
            "Eve should use 'eve' adapter"
        )
    
    def test_all_use_same_base_model(self):
        """Test that all agents use the same underlying base model."""
        self.assertIs(
            self.model_manager.alice.base_model,
            self.model_manager.eve.base_model,
            "Alice and Eve should share the same base_model instance"
        )
    
    def test_alice_bob_identical_responses_greedy(self):
        """Test that Alice and Bob produce identical responses with greedy decoding."""
        test_prompts = [
            "Say hello:",
            "Count to 5:",
            "What is 2+2?",
        ]
        
        for prompt in test_prompts:
            with self.subTest(prompt=prompt):
                alice_response = self.model_manager.alice.generate(
                    prompt,
                    max_new_tokens=20,
                    do_sample=False  # Greedy decoding for determinism
                )
                bob_response = self.model_manager.bob.generate(
                    prompt,
                    max_new_tokens=20,
                    do_sample=False  # Greedy decoding for determinism
                )
                
                self.assertEqual(
                    alice_response,
                    bob_response,
                    f"Alice and Bob should give identical responses for prompt: '{prompt}'\n"
                    f"Alice: {alice_response}\n"
                    f"Bob: {bob_response}"
                )
    
    def test_alice_bob_identical_responses_deterministic_sampling(self):
        """Test that Alice and Bob produce identical responses with low temperature sampling."""
        # Use very low temperature for near-deterministic sampling
        prompt = "The answer is:"
        
        alice_response = self.model_manager.alice.generate(
            prompt,
            max_new_tokens=15,
            temperature=0.1,  # Very low temperature
            do_sample=True
        )
        bob_response = self.model_manager.bob.generate(
            prompt,
            max_new_tokens=15,
            temperature=0.1,  # Very low temperature
            do_sample=True
        )
        
        # With very low temperature, responses should be identical or very similar
        # We allow for potential minor variations due to randomness
        self.assertEqual(
            alice_response,
            bob_response,
            f"Alice and Bob should give identical responses with low temperature\n"
            f"Alice: {alice_response}\n"
            f"Bob: {bob_response}\n"
            f"Note: If this test fails intermittently, it may be due to sampling randomness"
        )
    
    def test_eve_different_adapter(self):
        """Test that Eve uses a different adapter than Alice/Bob."""
        # Eve should have a different adapter, but since weights are copied at init,
        # we just verify the adapter name is different
        self.assertNotEqual(
            self.model_manager.eve.adapter_name,
            self.model_manager.alice.adapter_name,
            "Eve should use a different adapter name than Alice/Bob"
        )
    
    def test_model_info(self):
        """Test that model info reports correct configuration."""
        model_info = self.model_manager.get_model_info()
        
        self.assertTrue(model_info['alice_bob_shared'], "alice_bob_shared should be True")
        self.assertEqual(model_info['num_adapters'], 2, "Should have 2 adapters in shared mode")
        self.assertEqual(model_info['quantization'], "4-bit", "Should use 4-bit quantization")
        self.assertGreater(model_info['trainable_parameters'], 0, "Should have trainable parameters")


class TestSeparateAdapters(unittest.TestCase):
    """Test that Alice and Bob have separate adapters when shared_alice_bob=False."""
    
    @classmethod
    def setUpClass(cls):
        """Load models once for all tests."""
        cls.model_config = ModelConfig(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            shared_alice_bob=False,  # Separate adapters
            use_4bit=True,
            lora_r=64,
            max_new_tokens=128,
            temperature=0.7,
        )
        
        print(f"\nLoading models with SEPARATE adapters")
        print("This may take 2-5 minutes...\n")
        
        cls.model_manager = MultiAgentModelManager(cls.model_config)
        cls.model_manager.initialize()
        
        print("Models loaded successfully!\n")
    
    def test_alice_bob_different_objects(self):
        """Test that Alice and Bob are different objects when shared_alice_bob=False."""
        self.assertIsNot(
            self.model_manager.alice,
            self.model_manager.bob,
            "Alice and Bob should be different objects when shared_alice_bob=False"
        )
    
    def test_adapter_names_separate(self):
        """Test that adapter names are unique for each agent."""
        self.assertEqual(self.model_manager.alice.adapter_name, "alice")
        self.assertEqual(self.model_manager.bob.adapter_name, "bob")
        self.assertEqual(self.model_manager.eve.adapter_name, "eve")
        
        # All should be different
        adapter_names = {
            self.model_manager.alice.adapter_name,
            self.model_manager.bob.adapter_name,
            self.model_manager.eve.adapter_name,
        }
        self.assertEqual(len(adapter_names), 3, "All three agents should have unique adapter names")
    
    def test_all_use_same_base_model(self):
        """Test that all agents still use the same underlying base model."""
        self.assertIs(
            self.model_manager.alice.base_model,
            self.model_manager.bob.base_model,
            "Alice and Bob should share the same base_model instance"
        )
        self.assertIs(
            self.model_manager.bob.base_model,
            self.model_manager.eve.base_model,
            "Bob and Eve should share the same base_model instance"
        )
    
    def test_model_info_separate(self):
        """Test that model info reports correct configuration for separate mode."""
        model_info = self.model_manager.get_model_info()
        
        self.assertFalse(model_info['alice_bob_shared'], "alice_bob_shared should be False")
        self.assertEqual(model_info['num_adapters'], 3, "Should have 3 adapters in separate mode")


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    
    # Add shared adapter tests
    suite.addTest(unittest.makeSuite(TestSharedAdapter))
    
    # Add separate adapter tests (optional - takes extra time)
    # Uncomment to test both modes:
    # suite.addTest(unittest.makeSuite(TestSeparateAdapters))
    
    return suite


if __name__ == '__main__':
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
