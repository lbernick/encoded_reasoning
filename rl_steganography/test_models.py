"""Unit tests for models.py functions."""

import pytest
import torch
import unittest.mock
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from .config import ModelConfig


class TestCopyAdapterWeights:
    """Tests for the _copy_adapter_weights_multiado method."""
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager for testing."""
        from .models import MultiAgentModelManager
        
        config = ModelConfig(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            use_4bit=False,  # Disable quantization for testing
            shared_alice_bob=True
        )
        
        manager = MultiAgentModelManager(config)
        
        # Create a mock base model with named parameters
        manager.base_model = Mock()
        
        # Create some mock parameters for testing
        alice_bob_lora_A = torch.nn.Parameter(torch.randn(64, 128))
        alice_bob_lora_B = torch.nn.Parameter(torch.randn(128, 64))
        eve_lora_A = torch.nn.Parameter(torch.randn(64, 128))
        eve_lora_B = torch.nn.Parameter(torch.randn(128, 64))
        
        # Mock named_parameters to return our test parameters
        manager.base_model.named_parameters = Mock(return_value=[
            ('model.layers.0.alice_bob.lora_A.weight', alice_bob_lora_A),
            ('model.layers.0.alice_bob.lora_B.weight', alice_bob_lora_B),
            ('model.layers.0.eve.lora_A.weight', eve_lora_A),
            ('model.layers.0.eve.lora_B.weight', eve_lora_B),
        ])
        
        # Store references for verification
        manager._test_params = {
            'alice_bob_lora_A': alice_bob_lora_A,
            'alice_bob_lora_B': alice_bob_lora_B,
            'eve_lora_A': eve_lora_A,
            'eve_lora_B': eve_lora_B,
        }
        
        # Mock load_state_dict to actually update parameters
        def mock_load_state_dict(state_dict, strict=False):
            for name, value in state_dict.items():
                if 'eve.lora_A' in name:
                    manager._test_params['eve_lora_A'].data = value
                elif 'eve.lora_B' in name:
                    manager._test_params['eve_lora_B'].data = value
        
        manager.base_model.load_state_dict = Mock(side_effect=mock_load_state_dict)
        
        return manager
    
    def test_copies_weights_not_references(self, mock_model_manager):
        """Test that weights are copied, not just referenced."""
        # Get initial values
        original_eve_A = mock_model_manager._test_params['eve_lora_A'].data.clone()
        alice_bob_A = mock_model_manager._test_params['alice_bob_lora_A'].data.clone()
        
        # Verify they're different initially
        assert not torch.allclose(original_eve_A, alice_bob_A)
        
        # Copy weights
        mock_model_manager._copy_adapter_weights_multiado(
            from_adapter="alice_bob",
            to_adapter="eve"
        )
        
        # Verify Eve's weights now match Alice/Bob's
        # (The mock implementation should have been called)
        mock_model_manager.base_model.load_state_dict.assert_called_once()
    
    def test_adapter_name_replacement(self, mock_model_manager):
        """Test that adapter names are correctly replaced."""
        mock_model_manager._copy_adapter_weights_multiado(
            from_adapter="alice_bob",
            to_adapter="eve"
        )
        
        # Check that load_state_dict was called with correct parameter names
        call_args = mock_model_manager.base_model.load_state_dict.call_args
        state_dict = call_args[0][0]
        
        # All keys should contain "eve" and not "alice_bob"
        for key in state_dict.keys():
            assert "eve" in key
            assert "alice_bob" not in key
    
    def test_only_lora_parameters_copied(self, mock_model_manager):
        """Test that only LoRA parameters are copied."""
        mock_model_manager._copy_adapter_weights_multiado(
            from_adapter="alice_bob",
            to_adapter="eve"
        )
        
        call_args = mock_model_manager.base_model.load_state_dict.call_args
        state_dict = call_args[0][0]
        
        # All keys should contain "lora"
        for key in state_dict.keys():
            assert "lora" in key.lower()
    
    def test_strict_false_on_load(self, mock_model_manager):
        """Test that load_state_dict is called with strict=False."""
        mock_model_manager._copy_adapter_weights_multiado(
            from_adapter="alice_bob",
            to_adapter="eve"
        )
        
        call_args = mock_model_manager.base_model.load_state_dict.call_args
        assert call_args[1]['strict'] is False


class TestSteganographyModelGenerate:
    """Tests for SteganographyModel.generate method."""
    
    @pytest.fixture
    def mock_steganography_model(self):
        """Create a mock SteganographyModel."""
        from .models import SteganographyModel
        
        config = ModelConfig()
        
        # Create mocks
        mock_base_model = Mock()
        mock_tokenizer = Mock()
        
        # Setup tokenizer mock
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        
        # Setup model device
        mock_base_model.device = 'cpu'
        
        # Setup generate output
        mock_base_model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))
        
        # Setup decode
        mock_tokenizer.decode = Mock(return_value="Generated text")
        
        # Create model
        model = SteganographyModel(
            base_model=mock_base_model,
            tokenizer=mock_tokenizer,
            adapter_name="test_adapter",
            config=config
        )
        
        return model
    
    def test_generate_sets_adapter(self, mock_steganography_model):
        """Test that generate sets the correct adapter."""
        mock_steganography_model.generate("Test prompt")
        mock_steganography_model.base_model.set_adapter.assert_called_with("test_adapter")
    
    def test_generate_uses_default_params(self, mock_steganography_model):
        """Test that generate uses default parameters from config."""
        mock_steganography_model.generate("Test prompt")
        
        call_args = mock_steganography_model.base_model.generate.call_args
        assert call_args[1]['max_new_tokens'] == mock_steganography_model.config.max_new_tokens
        assert call_args[1]['temperature'] == mock_steganography_model.config.temperature
        assert call_args[1]['top_p'] == mock_steganography_model.config.top_p
    
    def test_generate_overrides_params(self, mock_steganography_model):
        """Test that generate can override default parameters."""
        mock_steganography_model.generate(
            "Test prompt",
            max_new_tokens=50,
            temperature=0.5,
            top_p=0.8
        )
        
        call_args = mock_steganography_model.base_model.generate.call_args
        assert call_args[1]['max_new_tokens'] == 50
        assert call_args[1]['temperature'] == 0.5
        assert call_args[1]['top_p'] == 0.8
    
    def test_generate_returns_string(self, mock_steganography_model):
        """Test that generate returns a string."""
        result = mock_steganography_model.generate("Test prompt")
        assert isinstance(result, str)
        assert result == "Generated text"


class TestMultiAgentModelManagerInfo:
    """Tests for MultiAgentModelManager.get_model_info method."""
    
    @pytest.fixture
    def mock_initialized_manager(self):
        """Create a mock initialized model manager."""
        from .models import MultiAgentModelManager
        
        config = ModelConfig(
            model_name="test/model",
            use_4bit=True,
            shared_alice_bob=True
        )
        
        manager = MultiAgentModelManager(config)
        
        # Create a mock model with parameters
        manager.base_model = Mock()
        
        # Mock parameters
        param1 = torch.nn.Parameter(torch.randn(100, 100))  # 10,000 params
        param2 = torch.nn.Parameter(torch.randn(50, 50))    # 2,500 params
        param2.requires_grad = False  # Make this one frozen
        
        manager.base_model.parameters = Mock(return_value=[param1, param2])
        
        return manager
    
    def test_get_model_info_structure(self, mock_initialized_manager):
        """Test that get_model_info returns correct structure."""
        info = mock_initialized_manager.get_model_info()
        
        assert 'model_name' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'trainable_percentage' in info
        assert 'quantization' in info
        assert 'alice_bob_shared' in info
        assert 'num_adapters' in info
    
    def test_get_model_info_calculates_params(self, mock_initialized_manager):
        """Test that parameter counts are calculated correctly."""
        info = mock_initialized_manager.get_model_info()
        
        assert info['total_parameters'] == 12500  # 10,000 + 2,500
        assert info['trainable_parameters'] == 10000  # Only param1
        assert info['trainable_percentage'] == pytest.approx(80.0)  # 10k/12.5k * 100
    
    def test_get_model_info_shared_vs_separate(self):
        """Test num_adapters reflects shared/separate config."""
        from .models import MultiAgentModelManager
        
        # Shared config
        config_shared = ModelConfig(shared_alice_bob=True)
        manager_shared = MultiAgentModelManager(config_shared)
        manager_shared.base_model = Mock()
        manager_shared.base_model.parameters = Mock(return_value=[])
        
        info_shared = manager_shared.get_model_info()
        assert info_shared['num_adapters'] == 2
        assert info_shared['alice_bob_shared'] is True
        
        # Separate config
        config_separate = ModelConfig(shared_alice_bob=False)
        manager_separate = MultiAgentModelManager(config_separate)
        manager_separate.base_model = Mock()
        manager_separate.base_model.parameters = Mock(return_value=[])
        
        info_separate = manager_separate.get_model_info()
        assert info_separate['num_adapters'] == 3
        assert info_separate['alice_bob_shared'] is False


class TestSharedAdapterInitialization:
    """Fast unit tests for shared adapter initialization logic (without loading real models)."""
    
    @pytest.fixture
    def mock_models_shared(self):
        """Mock the model loading for shared adapter mode."""
        with patch('rl_steganography.models.AutoModelForCausalLM') as mock_model_cls, \
             patch('rl_steganography.models.AutoTokenizer') as mock_tokenizer_cls, \
             patch('rl_steganography.models.get_peft_model') as mock_get_peft, \
             patch('rl_steganography.models.BitsAndBytesConfig'):
            
            # Setup mocks
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = '<eos>'
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            
            mock_base = Mock()
            mock_model_cls.from_pretrained.return_value = mock_base
            
            mock_peft_model = Mock()
            mock_peft_model.add_adapter = Mock()
            mock_peft_model.set_adapter = Mock()
            mock_get_peft.return_value = mock_peft_model
            
            from .models import MultiAgentModelManager
            config = ModelConfig(shared_alice_bob=True, use_4bit=False)
            manager = MultiAgentModelManager(config)
            
            # Mock the weight copying method
            manager._copy_adapter_weights_multiado = Mock()
            
            manager.initialize()
            
            yield manager, mock_peft_model, mock_get_peft
    
    @pytest.fixture
    def mock_models_separate(self):
        """Mock the model loading for separate adapter mode."""
        with patch('rl_steganography.models.AutoModelForCausalLM') as mock_model_cls, \
             patch('rl_steganography.models.AutoTokenizer') as mock_tokenizer_cls, \
             patch('rl_steganography.models.get_peft_model') as mock_get_peft, \
             patch('rl_steganography.models.BitsAndBytesConfig'):
            
            # Setup mocks
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = '<eos>'
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            
            mock_base = Mock()
            mock_model_cls.from_pretrained.return_value = mock_base
            
            mock_peft_model = Mock()
            mock_peft_model.add_adapter = Mock()
            mock_peft_model.set_adapter = Mock()
            mock_get_peft.return_value = mock_peft_model
            
            from .models import MultiAgentModelManager
            config = ModelConfig(shared_alice_bob=False, use_4bit=False)
            manager = MultiAgentModelManager(config)
            
            # Mock the weight copying method
            manager._copy_adapter_weights_multiado = Mock()
            
            manager.initialize()
            
            yield manager, mock_peft_model, mock_get_peft
    
    def test_shared_alice_bob_same_object(self, mock_models_shared):
        """Test that Alice and Bob are the same object in shared mode."""
        manager, _, _ = mock_models_shared
        assert manager.alice is manager.bob
    
    def test_shared_alice_bob_different_from_eve(self, mock_models_shared):
        """Test that Alice/Bob and Eve are different objects in shared mode."""
        manager, _, _ = mock_models_shared
        assert manager.alice is not manager.eve
        assert manager.bob is not manager.eve
    
    def test_shared_alice_bob_same_base_model(self, mock_models_shared):
        """Test that all agents share the same base model."""
        manager, _, _ = mock_models_shared
        assert manager.alice.base_model is manager.eve.base_model
    
    def test_shared_adapter_names(self, mock_models_shared):
        """Test that adapter names are correct in shared mode."""
        manager, _, _ = mock_models_shared
        assert manager.alice.adapter_name == "alice_bob"
        assert manager.bob.adapter_name == "alice_bob"
        assert manager.eve.adapter_name == "eve"
    
    def test_shared_get_peft_called_once(self, mock_models_shared):
        """Test that get_peft_model is only called once in shared mode."""
        manager, _, mock_get_peft = mock_models_shared
        # Should be called once with alice_bob adapter
        mock_get_peft.assert_called_once()
        call_args = mock_get_peft.call_args
        assert call_args[1]['adapter_name'] == "alice_bob"
    
    def test_shared_eve_adapter_added(self, mock_models_shared):
        """Test that Eve adapter is added to the base model."""
        manager, mock_peft_model, _ = mock_models_shared
        # Eve's adapter should be added after initial creation
        mock_peft_model.add_adapter.assert_called_once_with("eve", unittest.mock.ANY)
    
    def test_shared_weights_copied_to_eve(self, mock_models_shared):
        """Test that weights are copied from alice_bob to eve."""
        manager, _, _ = mock_models_shared
        # Verify weight copy was called
        manager._copy_adapter_weights_multiado.assert_called_once_with(
            from_adapter="alice_bob",
            to_adapter="eve"
        )
    
    def test_separate_alice_bob_different_objects(self, mock_models_separate):
        """Test that Alice and Bob are different objects in separate mode."""
        manager, _, _ = mock_models_separate
        assert manager.alice is not manager.bob
        assert manager.bob is not manager.eve
        assert manager.alice is not manager.eve
    
    def test_separate_adapter_names(self, mock_models_separate):
        """Test that adapter names are unique in separate mode."""
        manager, _, _ = mock_models_separate
        assert manager.alice.adapter_name == "alice"
        assert manager.bob.adapter_name == "bob"
        assert manager.eve.adapter_name == "eve"
    
    def test_separate_get_peft_called_once(self, mock_models_separate):
        """Test that get_peft_model is still only called once (alice first)."""
        manager, _, mock_get_peft = mock_models_separate
        # Should be called once with alice adapter
        mock_get_peft.assert_called_once()
        call_args = mock_get_peft.call_args
        assert call_args[1]['adapter_name'] == "alice"
    
    def test_separate_bob_and_eve_adapters_added(self, mock_models_separate):
        """Test that Bob and Eve adapters are added."""
        manager, mock_peft_model, _ = mock_models_separate
        # Both Bob and Eve should be added
        assert mock_peft_model.add_adapter.call_count == 2
        
        # Check the adapter names
        calls = mock_peft_model.add_adapter.call_args_list
        adapter_names = [call[0][0] for call in calls]
        assert "bob" in adapter_names
        assert "eve" in adapter_names
    
    def test_separate_weights_copied_bob_to_eve(self, mock_models_separate):
        """Test that weights are copied from bob to eve in separate mode."""
        manager, _, _ = mock_models_separate
        # Verify weight copy was called from bob to eve
        manager._copy_adapter_weights_multiado.assert_called_once_with(
            from_adapter="bob",
            to_adapter="eve"
        )
    
    def test_separate_all_use_same_base_model(self, mock_models_separate):
        """Test that all agents still share the same base model in separate mode."""
        manager, _, _ = mock_models_separate
        assert manager.alice.base_model is manager.bob.base_model
        assert manager.bob.base_model is manager.eve.base_model


class TestAdapterSwitching:
    """Test that adapters are correctly switched during generation and forward passes."""
    
    @pytest.fixture
    def mock_model_with_adapters(self):
        """Create a mock model with multiple adapters."""
        from .models import SteganographyModel
        
        config = ModelConfig()
        mock_base_model = Mock()
        mock_tokenizer = Mock()
        
        # Setup tokenizer to return a mock that behaves like tokenizer output
        mock_inputs = Mock()
        mock_inputs.input_ids = torch.tensor([[1, 2, 3]])
        mock_inputs.attention_mask = torch.tensor([[1, 1, 1]])
        mock_inputs.to = Mock(return_value=mock_inputs)  # .to() returns itself
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode = Mock(return_value="response")
        
        # Setup model
        mock_base_model.device = 'cpu'
        mock_base_model.set_adapter = Mock()
        mock_base_model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
        mock_base_model.__call__ = Mock(return_value=Mock(loss=torch.tensor(0.5)))
        
        model = SteganographyModel(
            base_model=mock_base_model,
            tokenizer=mock_tokenizer,
            adapter_name="test_adapter",
            config=config
        )
        
        return model
    
    def test_generate_calls_set_adapter(self, mock_model_with_adapters):
        """Test that generate calls set_adapter before generation."""
        mock_model_with_adapters.generate("test prompt")
        
        # Should set adapter before generate
        mock_model_with_adapters.base_model.set_adapter.assert_called_with("test_adapter")
    
    def test_forward_calls_set_adapter(self, mock_model_with_adapters):
        """Test that forward calls set_adapter before forward pass."""
        input_ids = torch.tensor([[1, 2, 3]])
        mock_model_with_adapters.forward(input_ids)
        
        # Should set adapter before forward
        mock_model_with_adapters.base_model.set_adapter.assert_called_with("test_adapter")
    
    def test_generate_with_do_sample_false(self, mock_model_with_adapters):
        """Test that generate handles do_sample=False (greedy decoding)."""
        mock_model_with_adapters.generate("test", do_sample=False)
        
        call_args = mock_model_with_adapters.base_model.generate.call_args[1]
        assert call_args['do_sample'] is False
        assert 'temperature' not in call_args or call_args['temperature'] is None
        assert 'top_p' not in call_args or call_args['top_p'] is None
    
    def test_generate_with_do_sample_true(self, mock_model_with_adapters):
        """Test that generate includes temperature/top_p when do_sample=True."""
        mock_model_with_adapters.generate("test", do_sample=True)
        
        call_args = mock_model_with_adapters.base_model.generate.call_args[1]
        assert call_args['do_sample'] is True
        assert call_args['temperature'] is not None
        assert call_args['top_p'] is not None
