from __future__ import annotations

import argparse
from typing import Any

import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .dataset import DatasetBuildConfig, build_hf_sft_dataset
from .experiment import DEFAULT_EXPERIMENT, describe_experiment


def _validate_chat_template(tokenizer) -> None:
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise ValueError(
            "Tokenizer has no chat template. "
            "Use an instruct/chat model with a chat_template."
        )


def _tokenize_example(
    example: dict[str, Any],
    tokenizer,
    max_length: int | None,
) -> dict[str, Any]:
    messages = example["messages"]
    _validate_chat_template(tokenizer)

    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )

    input_ids = list(full_ids)
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]

    if max_length is not None and len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def _tokenize_dataset(
    dataset: HFDataset, tokenizer, max_length: int | None
) -> HFDataset:
    return dataset.map(
        lambda ex: _tokenize_example(ex, tokenizer, max_length),
        remove_columns=dataset.column_names,
    )


def _load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list[str],
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=tuple(lora_target_modules),
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emoji-only reasoning SFT (HF local)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--output-dir", default="./outputs/sft_emoji_gsm8k")
    parser.add_argument("--seed", type=int, default=42)

    # Dataset
    parser.add_argument("--dataset-name", default=DEFAULT_EXPERIMENT.dataset_name)
    parser.add_argument("--dataset-split", default=DEFAULT_EXPERIMENT.dataset_split)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--distilled-path",
        default=None,
        help="Path to distilled JSONL (uses emoji_reasoning from file instead of template)",
    )

    # Prompt/constraint
    parser.add_argument("--constraint-name", default=DEFAULT_EXPERIMENT.constraint_name)
    parser.add_argument(
        "--emoji-reasoning-template",
        default=DEFAULT_EXPERIMENT.emoji_reasoning_template,
    )
    parser.add_argument(
        "--include-example",
        action="store_true",
        default=DEFAULT_EXPERIMENT.include_example,
    )

    # Training
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)

    # LoRA / QLoRA
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="use_4bit", action="store_false")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    print(describe_experiment(DEFAULT_EXPERIMENT))
    print("\nResolved args:")
    for key, value in sorted(vars(args).items()):
        print(f"- {key}: {value}")
    print()

    dataset_cfg = DatasetBuildConfig(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        seed=args.seed,
        max_samples=args.max_samples,
        constraint_name=args.constraint_name,
        include_example=args.include_example,
        emoji_reasoning_template=args.emoji_reasoning_template,
        distilled_path=args.distilled_path,
    )
    train_dataset = build_hf_sft_dataset(dataset_cfg)

    model, tokenizer = _load_model_and_tokenizer(
        model_name=args.model,
        use_4bit=args.use_4bit,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )

    tokenized = _tokenize_dataset(
        dataset=train_dataset, tokenizer=tokenizer, max_length=args.max_seq_length
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=[],
        bf16=True,
    )

    # TODO: Consider TRL SFTTrainer if we need packing or formatting changes.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
