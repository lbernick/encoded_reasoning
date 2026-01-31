"""Model loading and adapter management for RL steganography."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional, Literal
import logging
from pathlib import Path

from .config import ModelConfig

logger = logging.getLogger(__name__)

AgentType = Literal["alice", "bob", "eve"]


class SteganographyModel:
    """Wrapper for a model with LoRA adapter for steganography agents."""
    
    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        adapter_name: str,
        config: ModelConfig,
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.adapter_name = adapter_name
        self.config = config
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text from the model."""
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode only the generated tokens (not the prompt)
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model."""
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def save_adapter(self, save_path: str):
        """Save the LoRA adapter."""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(save_path)
        logger.info(f"Saved {self.adapter_name} adapter to {save_path}")
    
    def load_adapter(self, load_path: str):
        """Load a LoRA adapter."""
        self.base_model = PeftModel.from_pretrained(self.base_model, load_path)
        logger.info(f"Loaded {self.adapter_name} adapter from {load_path}")


class MultiAgentModelManager:
    """Manages the three agents (Alice, Bob, Eve) with shared base model."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.base_model = None
        self.alice = None
        self.bob = None
        self.eve = None
        
    def initialize(self):
        """Initialize the base model and create three adapters."""
        logger.info(f"Loading base model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization for Q-LoRA
        quantization_config = None
        if self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        logger.info("Creating LoRA adapters for Alice, Bob, and Eve")
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Create three separate adapters
        self.alice = self._create_agent_model("alice", lora_config)
        self.bob = self._create_agent_model("bob", lora_config)
        self.eve = self._create_agent_model("eve", lora_config)
        
        # Initialize Bob and Eve with same weights (evenly matched)
        logger.info("Initializing Bob and Eve with matching weights")
        self._copy_adapter_weights(source=self.bob, target=self.eve)
        
        logger.info("Model initialization complete")
        
    def _create_agent_model(self, agent_name: str, lora_config: LoraConfig) -> SteganographyModel:
        """Create a model with a LoRA adapter for an agent."""
        # Create a new adapter on the base model
        # Note: In a full implementation, we'd need to manage multiple adapters
        # For now, we create separate model instances
        model_with_adapter = get_peft_model(self.base_model, lora_config)
        
        return SteganographyModel(
            base_model=model_with_adapter,
            tokenizer=self.tokenizer,
            adapter_name=agent_name,
            config=self.config,
        )
    
    def _copy_adapter_weights(self, source: SteganographyModel, target: SteganographyModel):
        """Copy adapter weights from source to target."""
        source_state = source.base_model.state_dict()
        target.base_model.load_state_dict(source_state)
    
    def save_all_adapters(self, save_dir: str, episode: int):
        """Save all three adapters."""
        base_path = Path(save_dir) / f"episode_{episode}"
        self.alice.save_adapter(str(base_path / "alice"))
        self.bob.save_adapter(str(base_path / "bob"))
        self.eve.save_adapter(str(base_path / "eve"))
        logger.info(f"Saved all adapters at episode {episode}")
    
    def get_model_info(self) -> dict:
        """Get information about the models."""
        num_params = sum(p.numel() for p in self.base_model.parameters())
        num_trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name,
            "total_parameters": num_params,
            "trainable_parameters": num_trainable,
            "trainable_percentage": 100 * num_trainable / num_params,
            "quantization": "4-bit" if self.config.use_4bit else "none",
        }
