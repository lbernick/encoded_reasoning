from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import json
from pathlib import Path

from datasets import Dataset as HFDataset
from inspect_ai.dataset import Sample

from ..evals.constraints import get_constraint
from ..evals.datasets import (
    DatasetType,
    get_dataset_system_prompt,
    get_dataset_type,
    load_dataset,
)
from ..evals.runner import get_base_system_prompt, get_example


@dataclass(frozen=True)
class DatasetBuildConfig:
    dataset_name: str
    split: str = "train"
    seed: int = 42
    max_samples: int | None = None
    constraint_name: str = "only_emojis"
    include_example: bool = True
    emoji_reasoning_template: str = ""
    distilled_path: str | None = None


def build_system_prompt(
    dataset_name: str,
    constraint_name: str,
    include_example: bool = True,
) -> str:
    constraint = get_constraint(constraint_name)
    dataset_type: DatasetType | None = get_dataset_type(dataset_name)
    if dataset_type is None:
        raise ValueError(f"Dataset type not found for: {dataset_name}")

    base_prompt = get_base_system_prompt(reasoning=True, dataset_type=dataset_type)
    full_prompt = base_prompt + "\n" + constraint.system_prompt

    dataset_prompt = get_dataset_system_prompt(dataset_name)
    if dataset_prompt:
        full_prompt += "\n" + dataset_prompt

    if include_example:
        full_prompt += get_example(constraint.reasoning_example, dataset_type)

    return full_prompt


def _iter_samples(inspect_dataset: Iterable[Sample]) -> Iterable[Sample]:
    for sample in inspect_dataset:
        yield sample


def _load_distilled_rows(path: str | Path) -> list[dict[str, object]]:
    distilled_path = Path(path)
    if not distilled_path.exists():
        raise FileNotFoundError(f"Distilled JSONL not found: {distilled_path}")

    rows: list[dict[str, object]] = []
    with distilled_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} in {distilled_path}"
                ) from exc
    return rows


def _extract_distilled_fields(row: dict[str, object]) -> tuple[str, str, str]:
    question = row.get("question") or row.get("input")
    if not isinstance(question, str):
        raise ValueError("Distilled row missing string 'question'.")

    answer = row.get("answer") or row.get("gold_answer") or row.get("target")
    if not isinstance(answer, str):
        answer = str(answer) if answer is not None else None
    if not answer:
        raise ValueError("Distilled row missing 'answer' or 'gold_answer'.")

    emoji_reasoning = row.get("emoji_reasoning")
    if not isinstance(emoji_reasoning, str):
        raise ValueError("Distilled row missing string 'emoji_reasoning'.")

    return question, answer, emoji_reasoning


def build_hf_sft_dataset(config: DatasetBuildConfig) -> HFDataset:
    if config.distilled_path:
        distilled_rows = _load_distilled_rows(config.distilled_path)
        if config.max_samples is not None:
            distilled_rows = distilled_rows[: config.max_samples]
    else:
        inspect_dataset = load_dataset(
            config.dataset_name,
            split=config.split,
            shuffle=True,
            seed=config.seed,
        )

        if config.max_samples is not None:
            inspect_dataset = inspect_dataset[: config.max_samples]

    system_prompt = build_system_prompt(
        dataset_name=config.dataset_name,
        constraint_name=config.constraint_name,
        include_example=config.include_example,
    )

    rows = []
    if config.distilled_path:
        for row in distilled_rows:
            question, answer, emoji_reasoning = _extract_distilled_fields(row)
            assistant = (
                f"<reasoning>{emoji_reasoning}</reasoning>\n"
                f"<answer>{answer}</answer>"
            )
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": assistant},
                    ],
                    "question": question,
                    "answer": answer,
                    "emoji_reasoning": emoji_reasoning,
                    "distilled": row,
                }
            )
    else:
        for sample in _iter_samples(inspect_dataset):
            if not isinstance(sample.input, str):
                raise ValueError(
                    "Expected string inputs from evals datasets. "
                    "Got non-string sample.input."
                )
            question = sample.input
            answer = sample.target
            if not isinstance(answer, str):
                answer = str(answer)
            assistant = (
                f"<reasoning>{config.emoji_reasoning_template}</reasoning>\n"
                f"<answer>{answer}</answer>"
            )
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": assistant},
                    ],
                    "question": question,
                    "answer": answer,
                }
            )

    return HFDataset.from_list(rows)
