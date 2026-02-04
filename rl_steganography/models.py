"""Model loading and adapter management for RL steganography."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional, Literal, List
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

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Generate text from the model."""
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p

        # Default to sampling unless explicitly disabled
        if do_sample is None:
            do_sample = True

        # Set the active adapter for this model
        self.base_model.set_adapter(self.adapter_name)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs,
        }

        if stop_sequences:
            generation_kwargs["stop_strings"] = stop_sequences
            generation_kwargs["tokenizer"] = self.tokenizer

        outputs = self.base_model.generate(**inputs, **generation_kwargs)

        # Decode only the generated tokens (not the prompt)
        generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        # Trim any stop sequences from the end of the string
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)

        return generated_text.strip()

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model."""
        # Set the active adapter for this model
        self.base_model.set_adapter(self.adapter_name)

        return self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
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
    """Manages the agents (Alice, Bob, Eve) with shared base model."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.base_model = None
        self.alice = None
        self.bob = None
        self.eve = None
        self.alice_bob_shared = config.shared_alice_bob

    def initialize(self):
        """Initialize the base model and create adapters."""
        logger.info(f"Loading base model: {self.config.model_name}")
        logger.info(
            f"Alice/Bob configuration: {'SHARED adapter' if self.alice_bob_shared else 'SEPARATE adapters'}"
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
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
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Create LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if self.alice_bob_shared:
            # Alice and Bob share the same adapter, Eve has separate
            logger.info("Creating 2 LoRA adapters: Alice/Bob (shared) and Eve")

            # Create model with Alice/Bob adapter
            self.base_model = get_peft_model(
                base_model, lora_config, adapter_name="alice_bob"
            )
            self.alice = SteganographyModel(
                base_model=self.base_model,
                tokenizer=self.tokenizer,
                adapter_name="alice_bob",
                config=self.config,
            )
            self.bob = self.alice  # Bob references the same model

            # Add Eve adapter to the same base model
            self.base_model.add_adapter("eve", lora_config)

            # Create Eve wrapper that uses the eve adapter
            self.eve = SteganographyModel(
                base_model=self.base_model,
                tokenizer=self.tokenizer,
                adapter_name="eve",
                config=self.config,
            )

            # Initialize Eve with same weights as Alice/Bob (evenly matched)
            logger.info("Initializing Eve with matching weights to Alice/Bob")
            self._copy_adapter_weights_multiado(
                from_adapter="alice_bob", to_adapter="eve"
            )
        else:
            # Alice, Bob, and Eve have separate adapters
            logger.info("Creating 3 LoRA adapters: Alice, Bob, and Eve")

            # Create model with Alice adapter
            self.base_model = get_peft_model(
                base_model, lora_config, adapter_name="alice"
            )
            self.alice = SteganographyModel(
                base_model=self.base_model,
                tokenizer=self.tokenizer,
                adapter_name="alice",
                config=self.config,
            )

            # Add Bob adapter
            self.base_model.add_adapter("bob", lora_config)
            self.bob = SteganographyModel(
                base_model=self.base_model,
                tokenizer=self.tokenizer,
                adapter_name="bob",
                config=self.config,
            )

            # Add Eve adapter
            self.base_model.add_adapter("eve", lora_config)
            self.eve = SteganographyModel(
                base_model=self.base_model,
                tokenizer=self.tokenizer,
                adapter_name="eve",
                config=self.config,
            )

            # Initialize Bob and Eve with same weights (evenly matched)
            logger.info("Initializing Bob and Eve with matching weights")
            self._copy_adapter_weights_multiado(from_adapter="bob", to_adapter="eve")

        logger.info("Model initialization complete")

    def _create_agent_model(
        self, agent_name: str, lora_config: LoraConfig
    ) -> SteganographyModel:
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

    def _copy_adapter_weights_multiado(self, from_adapter: str, to_adapter: str):
        """Copy adapter weights from one adapter to another in a multi-adapter setup."""
        # Get state dict for the source adapter
        source_state = {}
        for name, param in self.base_model.named_parameters():
            # Look for parameters belonging to the source adapter
            if from_adapter in name and "lora" in name.lower():
                # Create corresponding target adapter parameter name
                target_name = name.replace(from_adapter, to_adapter)
                source_state[target_name] = param.data.clone()

        # Load into target adapter
        self.base_model.load_state_dict(source_state, strict=False)

    def save_all_adapters(self, save_dir: str, episode: int):
        """Save all adapters."""
        base_path = Path(save_dir) / f"episode_{episode}"

        if self.alice_bob_shared:
            # Save Alice/Bob shared adapter and Eve
            self.alice.save_adapter(str(base_path / "alice_bob"))
            self.eve.save_adapter(str(base_path / "eve"))
            logger.info(f"Saved alice_bob and eve adapters at episode {episode}")
        else:
            # Save all three separate adapters
            self.alice.save_adapter(str(base_path / "alice"))
            self.bob.save_adapter(str(base_path / "bob"))
            self.eve.save_adapter(str(base_path / "eve"))
            logger.info(f"Saved alice, bob, and eve adapters at episode {episode}")

    def get_model_info(self) -> dict:
        """Get information about the models."""
        num_params = sum(p.numel() for p in self.base_model.parameters())
        num_trainable = sum(
            p.numel() for p in self.base_model.parameters() if p.requires_grad
        )

        return {
            "model_name": self.config.model_name,
            "total_parameters": num_params,
            "trainable_parameters": num_trainable,
            "trainable_percentage": 100 * num_trainable / num_params,
            "quantization": "4-bit" if self.config.use_4bit else "none",
            "alice_bob_shared": self.alice_bob_shared,
            "num_adapters": 2 if self.alice_bob_shared else 3,
        }
