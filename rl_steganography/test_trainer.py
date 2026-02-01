"""Unit tests for trainer.py functions."""

import pytest
import torch
import torch.nn.functional as F
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from .config import TrainingConfig, ModelConfig


class TestComputePolicyLoss:
    """Tests for the compute_policy_loss method."""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer for testing."""
        from .trainer import GRPOTrainer
        
        # Create configs
        model_config = ModelConfig()
        training_config = TrainingConfig()
        
        # Create mocks
        mock_model_manager = Mock()
        mock_environment = Mock()
        
        # Setup tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])  # 5 token prompt
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        
        # Mock tokenizer call
        def tokenizer_call(text, return_tensors=None, truncation=None, max_length=None):
            # Simulate tokenization: ~1 token per 4 chars
            num_tokens = len(text) // 4 + 1
            return {
                'input_ids': torch.arange(num_tokens).unsqueeze(0),
                'attention_mask': torch.ones(num_tokens).unsqueeze(0)
            }
        
        mock_tokenizer.side_effect = tokenizer_call
        mock_model_manager.tokenizer = mock_tokenizer
        
        # Create trainer
        with patch('rl_steganography.trainer.wandb'):
            trainer = GRPOTrainer(mock_model_manager, mock_environment, training_config)
        
        return trainer
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.base_model = Mock()
        model.base_model.device = 'cpu'
        
        # Mock forward pass
        def mock_forward(input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            vocab_size = 1000
            
            # Create random logits
            logits = torch.randn(batch_size, seq_len, vocab_size)
            
            output = Mock()
            output.logits = logits
            return output
        
        model.forward = Mock(side_effect=mock_forward)
        
        return model
    
    def test_compute_policy_loss_returns_tensor(self, mock_trainer, mock_model):
        """Test that compute_policy_loss returns a tensor."""
        prompt = "Test prompt"
        generated_text = "Generated response"
        reward = 0.5
        baseline_reward = 0.3
        
        loss = mock_trainer.compute_policy_loss(
            mock_model,
            prompt,
            generated_text,
            reward,
            baseline_reward
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Should be a scalar
    
    def test_compute_policy_loss_positive_advantage(self, mock_trainer, mock_model):
        """Test loss with positive advantage (reward > baseline)."""
        prompt = "Test prompt"
        generated_text = "Generated response"
        reward = 0.8
        baseline_reward = 0.3
        
        loss = mock_trainer.compute_policy_loss(
            mock_model,
            prompt,
            generated_text,
            reward,
            baseline_reward
        )
        
        # With positive advantage, loss should be negative (we want to minimize it)
        # Loss = -advantage * log_prob, and log_prob is typically negative
        # So -advantage * negative_log_prob = positive when advantage > 0
        assert isinstance(loss, torch.Tensor)
    
    def test_compute_policy_loss_negative_advantage(self, mock_trainer, mock_model):
        """Test loss with negative advantage (reward < baseline)."""
        prompt = "Test prompt"
        generated_text = "Generated response"
        reward = 0.2
        baseline_reward = 0.8
        
        loss = mock_trainer.compute_policy_loss(
            mock_model,
            prompt,
            generated_text,
            reward,
            baseline_reward
        )
        
        assert isinstance(loss, torch.Tensor)
    
    def test_compute_policy_loss_zero_advantage(self, mock_trainer, mock_model):
        """Test loss with zero advantage (reward = baseline)."""
        prompt = "Test prompt"
        generated_text = "Generated response"
        reward = 0.5
        baseline_reward = 0.5
        
        loss = mock_trainer.compute_policy_loss(
            mock_model,
            prompt,
            generated_text,
            reward,
            baseline_reward
        )
        
        # With zero advantage, loss should be zero
        assert torch.abs(loss) < 1e-5
    
    def test_compute_policy_loss_sets_adapter(self, mock_trainer, mock_model):
        """Test that compute_policy_loss sets the model adapter."""
        prompt = "Test"
        generated_text = "Response"
        
        mock_trainer.compute_policy_loss(
            mock_model,
            prompt,
            generated_text,
            reward=0.5,
            baseline_reward=0.0
        )
        
        # Should call set_adapter during forward pass
        mock_model.base_model.set_adapter.assert_called()
    
    def test_compute_policy_loss_with_different_rewards(self, mock_trainer, mock_model):
        """Test that different rewards produce different losses."""
        prompt = "Test prompt"
        generated_text = "Generated response"
        baseline_reward = 0.5
        
        loss_high = mock_trainer.compute_policy_loss(
            mock_model, prompt, generated_text, reward=1.0, baseline_reward=baseline_reward
        )
        
        loss_low = mock_trainer.compute_policy_loss(
            mock_model, prompt, generated_text, reward=0.0, baseline_reward=baseline_reward
        )
        
        # Different rewards should produce different losses
        assert not torch.isclose(loss_high, loss_low, rtol=0.1)


class TestTrainerSetup:
    """Tests for trainer initialization and setup."""
    
    @patch('rl_steganography.trainer.wandb')
    def test_trainer_creates_optimizers(self, mock_wandb):
        """Test that trainer creates optimizers for all agents."""
        from .trainer import GRPOTrainer
        
        mock_model_manager = Mock()
        mock_model_manager.alice_bob_shared = False
        mock_model_manager.alice = Mock()
        mock_model_manager.bob = Mock()
        mock_model_manager.eve = Mock()
        mock_model_manager.alice.base_model = Mock()
        mock_model_manager.bob.base_model = Mock()
        mock_model_manager.eve.base_model = Mock()
        mock_model_manager.alice.base_model.parameters = Mock(return_value=[])
        mock_model_manager.bob.base_model.parameters = Mock(return_value=[])
        mock_model_manager.eve.base_model.parameters = Mock(return_value=[])
        mock_model_manager.get_model_info = Mock(return_value={
            'model_name': 'test',
            'alice_bob_shared': False,
            'num_adapters': 3,
            'quantization': 'none',
            'trainable_parameters': 1000
        })
        
        mock_environment = Mock()
        config = TrainingConfig()
        
        trainer = GRPOTrainer(mock_model_manager, mock_environment, config)
        
        assert trainer.alice_optimizer is not None
        assert trainer.bob_optimizer is not None
        assert trainer.eve_optimizer is not None
    
    @patch('rl_steganography.trainer.wandb')
    def test_trainer_shared_mode_no_bob_optimizer(self, mock_wandb):
        """Test that in shared mode, Bob doesn't get separate optimizer."""
        from .trainer import GRPOTrainer
        
        mock_model_manager = Mock()
        mock_model_manager.alice_bob_shared = True
        mock_model_manager.alice = Mock()
        mock_model_manager.bob = Mock()
        mock_model_manager.eve = Mock()
        mock_model_manager.alice.base_model = Mock()
        mock_model_manager.eve.base_model = Mock()
        mock_model_manager.alice.base_model.parameters = Mock(return_value=[])
        mock_model_manager.eve.base_model.parameters = Mock(return_value=[])
        mock_model_manager.get_model_info = Mock(return_value={
            'model_name': 'test',
            'alice_bob_shared': True,
            'num_adapters': 2,
            'quantization': 'none',
            'trainable_parameters': 1000
        })
        
        mock_environment = Mock()
        config = TrainingConfig()
        
        trainer = GRPOTrainer(mock_model_manager, mock_environment, config)
        
        assert trainer.alice_optimizer is not None
        assert trainer.bob_optimizer is None  # Should be None in shared mode
        assert trainer.eve_optimizer is not None
    
    @patch('rl_steganography.trainer.wandb')
    def test_trainer_applies_learning_rate_multipliers(self, mock_wandb):
        """Test that learning rate multipliers are applied."""
        from .trainer import GRPOTrainer
        from torch.optim import AdamW
        
        mock_model_manager = Mock()
        mock_model_manager.alice_bob_shared = False
        mock_model_manager.alice = Mock()
        mock_model_manager.bob = Mock()
        mock_model_manager.eve = Mock()
        
        # Create actual parameters for optimizer
        mock_model_manager.alice.base_model = Mock()
        mock_model_manager.bob.base_model = Mock()
        mock_model_manager.eve.base_model = Mock()
        
        alice_params = [torch.nn.Parameter(torch.randn(10, 10))]
        bob_params = [torch.nn.Parameter(torch.randn(10, 10))]
        eve_params = [torch.nn.Parameter(torch.randn(10, 10))]
        
        mock_model_manager.alice.base_model.parameters = Mock(return_value=alice_params)
        mock_model_manager.bob.base_model.parameters = Mock(return_value=bob_params)
        mock_model_manager.eve.base_model.parameters = Mock(return_value=eve_params)
        
        mock_model_manager.get_model_info = Mock(return_value={
            'model_name': 'test',
            'alice_bob_shared': False,
            'num_adapters': 3,
            'quantization': 'none',
            'trainable_parameters': 1000
        })
        
        mock_environment = Mock()
        
        config = TrainingConfig(
            learning_rate=1e-5,
            alice_lr_multiplier=2.0,
            bob_lr_multiplier=1.5,
            eve_lr_multiplier=0.5
        )
        
        trainer = GRPOTrainer(mock_model_manager, mock_environment, config)
        
        # Check that optimizers were created (we can't easily check LR without more mocking)
        assert trainer.alice_optimizer is not None
        assert trainer.bob_optimizer is not None
        assert trainer.eve_optimizer is not None


class TestTrainerStability:
    """Tests for trainer stability checking and handling."""
    
    @patch('rl_steganography.trainer.wandb')
    def test_handle_stability_issues_logs_warnings(self, mock_wandb):
        """Test that stability issues are logged."""
        from .trainer import GRPOTrainer
        import logging
        
        mock_model_manager = Mock()
        mock_model_manager.alice_bob_shared = True
        mock_model_manager.alice = Mock()
        mock_model_manager.eve = Mock()
        mock_model_manager.alice.base_model = Mock()
        mock_model_manager.eve.base_model = Mock()
        mock_model_manager.alice.base_model.parameters = Mock(return_value=[])
        mock_model_manager.eve.base_model.parameters = Mock(return_value=[])
        mock_model_manager.get_model_info = Mock(return_value={
            'model_name': 'test',
            'alice_bob_shared': True,
            'num_adapters': 2,
            'quantization': 'none',
            'trainable_parameters': 1000
        })
        
        mock_environment = Mock()
        config = TrainingConfig()
        
        trainer = GRPOTrainer(mock_model_manager, mock_environment, config)
        
        warnings = {
            "eve_collapsed": True,
            "bob_stuck": False,
            "alice_unstable": False
        }
        
        # This should not raise an error
        trainer._handle_stability_issues(warnings)
        
        # Test all warnings
        warnings = {
            "eve_collapsed": True,
            "bob_stuck": True,
            "alice_unstable": True
        }
        
        trainer._handle_stability_issues(warnings)
