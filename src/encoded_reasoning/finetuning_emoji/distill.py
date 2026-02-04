from __future__ import annotations

import argparse
import asyncio
import json
import gc
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable

import regex
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessorList,
)
from transformers.generation import StopStringCriteria

from ..evals.datasets import (
    DatasetType,
    extract_number_answer,
    get_dataset_system_prompt,
    get_dataset_type,
    load_dataset,
)
from ..evals.runner import get_base_system_prompt
from ..evals.token_filters import emoji_token_filter
from ..logit_masking.processor import MaskedReasoningProcessor


FALLBACK_EMOJI = "\U0001F914"  # 🤔
EMOJI_PATTERN = regex.compile(
    r"\p{Emoji_Presentation}|\p{Extended_Pictographic}|\p{Emoji_Component}"
)


@dataclass(frozen=True)
class DistillConfig:
    model: str
    dataset_name: str
    dataset_split: str
    seed: int
    max_samples: int | None
    output_dir: Path
    teacher_max_tokens: int
    emoji_max_tokens: int
    temperature: float | None
    top_p: float | None
    include_example: bool
    use_4bit: bool


def _build_teacher_prompt(dataset_name: str, include_example: bool) -> str:
    dataset_type: DatasetType | None = get_dataset_type(dataset_name)
    if dataset_type is None:
        raise ValueError(f"Dataset type not found for: {dataset_name}")
    base_prompt = get_base_system_prompt(reasoning=True, dataset_type=dataset_type)
    dataset_prompt = get_dataset_system_prompt(dataset_name)
    full_prompt = (
        base_prompt
        + "\nReturn output using <reasoning>...</reasoning><answer>...</answer> tags only."
    )
    if dataset_prompt:
        full_prompt += "\n" + dataset_prompt
    if include_example:
        full_prompt += "\nExample:\n<reasoning>\nStep-by-step work here...\n</reasoning>\n<answer>42</answer>"
    return full_prompt


def _build_emoji_prompt() -> str:
    return (
        "Rewrite the reasoning text as emojis only. Preserve order and meaning.\n"
        "Use only emojis, whitespace, and basic punctuation (.,;:!?). "
        "Do NOT use letters or digits.\n"
        "The prompt will end with <reasoning>; continue with emojis only, then close </reasoning>."
    )


def _extract_tag(text: str, tag: str) -> str | None:
    pattern = regex.compile(
        rf"<{tag}>(.*?)</{tag}>", regex.DOTALL | regex.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


def _count_emojis(text: str) -> int:
    return len(EMOJI_PATTERN.findall(text))


def _emoji_ratio(text: str) -> float:
    if not text:
        return 0.0
    return _count_emojis(text) / max(len(text), 1)


def _iter_samples(inspect_dataset: Iterable) -> Iterable:
    for sample in inspect_dataset:
        yield sample


def _load_local_model(model_name: str, use_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
    return tokenizer, model


def _generate_hf(
    tokenizer,
    model,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    stop_seqs: list[str] | None = None,
    logits_processor: LogitsProcessorList | None = None,
):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature is not None,
        "temperature": temperature,
        "top_p": top_p,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    stopping = None
    if stop_seqs:
        stopping = [StopStringCriteria(tokenizer, stop_seqs)]
        kwargs["stopping_criteria"] = stopping
    if logits_processor is not None:
        kwargs["logits_processor"] = logits_processor
    with torch.no_grad():
        outputs = model.generate(**inputs, **kwargs)
    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text


def _write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_run_dir(base_dir: Path, dataset_name: str, seed: int, max_samples: int | None) -> Path:
    suffix = f"n{max_samples}" if max_samples is not None else "all"
    return base_dir / f"distill_{dataset_name}_seed{seed}_{suffix}"


async def distill(config: DistillConfig) -> Path:
    run_dir = _build_run_dir(
        base_dir=config.output_dir,
        dataset_name=config.dataset_name,
        seed=config.seed,
        max_samples=config.max_samples,
    )
    teacher_path = run_dir / "teacher.jsonl"
    emoji_path = run_dir / "emoji.jsonl"
    distilled_path = run_dir / "distilled.jsonl"

    for path in (teacher_path, emoji_path, distilled_path):
        if path.exists():
            path.unlink()

    inspect_dataset = load_dataset(
        config.dataset_name,
        split=config.dataset_split,
        shuffle=True,
        seed=config.seed,
    )
    if config.max_samples is not None:
        inspect_dataset = islice(_iter_samples(inspect_dataset), config.max_samples)

    teacher_system_prompt = _build_teacher_prompt(
        config.dataset_name, config.include_example
    )
    emoji_system_prompt = _build_emoji_prompt()

    total = 0
    empty_emoji = 0
    emoji_ratios: list[float] = []

    print("Pass 1: teacher reasoning generation")
    teacher_tokenizer, teacher_model = _load_local_model(
        config.model, config.use_4bit
    )

    for idx, sample in enumerate(inspect_dataset):
        question = sample.input
        if not isinstance(question, str):
            raise ValueError("Expected string inputs from dataset.")
        gold_answer = sample.target
        if not isinstance(gold_answer, str):
            gold_answer = str(gold_answer)

        teacher_messages = [
            {"role": "system", "content": teacher_system_prompt},
            {"role": "user", "content": question},
        ]
        teacher_text = _generate_hf(
            tokenizer=teacher_tokenizer,
            model=teacher_model,
            messages=teacher_messages,
            max_tokens=config.teacher_max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        tag_reasoning = _extract_tag(teacher_text, "reasoning")
        if tag_reasoning:
            pre_tag = teacher_text.split("<reasoning>", 1)[0].strip()
            if pre_tag and len(pre_tag) > (len(tag_reasoning) * 2):
                teacher_reasoning = pre_tag
            else:
                teacher_reasoning = tag_reasoning
        else:
            teacher_reasoning = teacher_text.strip()
        teacher_answer = _extract_tag(teacher_text, "answer") or extract_number_answer(
            teacher_text
        )

        teacher_row = {
            "id": idx,
            "question": question,
            "gold_answer": gold_answer,
            "answer": gold_answer,
            "teacher_response": teacher_text,
            "teacher_reasoning": teacher_reasoning,
            "teacher_answer": teacher_answer,
            "teacher_prompt": {
                "system": teacher_system_prompt,
                "user": question,
            },
        }
        _write_jsonl(teacher_path, teacher_row)

        total += 1
        print(f"[{total}] distilled sample {idx}")

    del teacher_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nPass 2: emoji translation")
    emoji_tokenizer, emoji_model = _load_local_model(config.model, config.use_4bit)
    allowed_ids = emoji_token_filter(emoji_tokenizer)
    allowed_ids.add(emoji_tokenizer.eos_token_id)
    end_ids = emoji_tokenizer.encode("</reasoning>", add_special_tokens=False)
    processor = MaskedReasoningProcessor(
        tokenizer=emoji_tokenizer,
        allowed_ids=allowed_ids,
        vocab_size=emoji_model.config.vocab_size,
        end_ids=end_ids,
        max_masked_tokens=config.emoji_max_tokens,
        force_answer_prefix=None,
    )
    logits_processor = LogitsProcessorList([processor])

    with teacher_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            teacher_row = json.loads(line)
            idx = teacher_row.get("id", total)
            question = teacher_row.get("question", "")
            gold_answer = teacher_row.get("gold_answer", teacher_row.get("answer", ""))
            teacher_reasoning = teacher_row.get("teacher_reasoning") or teacher_row.get(
                "teacher_response", ""
            )

            emoji_messages = [
                {"role": "system", "content": emoji_system_prompt},
                {"role": "user", "content": f"{teacher_reasoning}\n\n<reasoning>"},
            ]
            emoji_text = _generate_hf(
                tokenizer=emoji_tokenizer,
                model=emoji_model,
                messages=emoji_messages,
                max_tokens=config.emoji_max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                stop_seqs=["</reasoning>"],
                logits_processor=logits_processor,
            )

            combined = f"<reasoning>{emoji_text}"
            emoji_reasoning = _extract_tag(combined, "reasoning") or emoji_text.strip()
            if "</reasoning>" in emoji_reasoning:
                emoji_reasoning = emoji_reasoning.replace("</reasoning>", "").strip()
            if _count_emojis(emoji_reasoning) == 0:
                emoji_reasoning = FALLBACK_EMOJI
                empty_emoji += 1

            ratio = _emoji_ratio(emoji_reasoning)
            emoji_ratios.append(ratio)

            emoji_row = {
                "id": idx,
                "question": question,
                "gold_answer": gold_answer,
                "answer": gold_answer,
                "teacher_reasoning": teacher_reasoning,
                "emoji_response": emoji_text,
                "emoji_reasoning": emoji_reasoning,
                "emoji_prompt": {
                    "system": emoji_system_prompt,
                    "user": teacher_reasoning,
                },
            }
            _write_jsonl(emoji_path, emoji_row)

            distilled_row = {
                "id": idx,
                "question": question,
                "gold_answer": gold_answer,
                "answer": gold_answer,
                "teacher_response": teacher_row.get("teacher_response", ""),
                "teacher_reasoning": teacher_reasoning,
                "teacher_answer": teacher_row.get("teacher_answer"),
                "emoji_response": emoji_text,
                "emoji_reasoning": emoji_reasoning,
                "teacher_prompt": teacher_row.get("teacher_prompt"),
                "emoji_prompt": emoji_row["emoji_prompt"],
                "metadata": {
                    "model": config.model,
                    "seed": config.seed,
                },
            }
            _write_jsonl(distilled_path, distilled_row)

    del emoji_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_ratio = sum(emoji_ratios) / max(len(emoji_ratios), 1)
    print("\nDistillation complete.")
    print(f"  Samples: {total}")
    print(f"  Empty emoji fallbacks: {empty_emoji}")
    print(f"  Avg emoji density (emoji chars / total chars): {avg_ratio:.3f}")
    print(f"  Output dir: {run_dir}")
    return distilled_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-pass emoji distillation (teacher -> emoji-only reasoning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="src/encoded_reasoning/finetuning_emoji/artifacts",
    )
    parser.add_argument("--teacher-max-tokens", type=int, default=768)
    parser.add_argument("--emoji-max-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=False,
        help="Load models in 4-bit to fit GPU memory",
    )
    parser.add_argument(
        "--include-example",
        action="store_true",
        default=False,
        help="Include a generic example in the teacher system prompt",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = DistillConfig(
        model=args.model,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        seed=args.seed,
        max_samples=args.max_samples,
        output_dir=Path(args.output_dir),
        teacher_max_tokens=args.teacher_max_tokens,
        emoji_max_tokens=args.emoji_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        include_example=args.include_example,
        use_4bit=args.use_4bit,
    )
    asyncio.run(distill(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
