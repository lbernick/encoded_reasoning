from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    """Top-level experiment spec for emoji-only reasoning SFT."""

    name: str = "emoji_only_gsm8k_sft"

    # Dataset
    dataset_name: str = "gsm8k"
    dataset_split: str = "train"
    max_samples: int | None = None

    # Prompting + constraints (reuse evals helpers)
    constraint_name: str = "only_emojis"
    include_example: bool = True

    # Target formatting (assistant output)
    emoji_reasoning_template: str = (
        "\U0001F914\U0001F9EE\U00002705"  # 🤔🧮✅
    )

    # Optimization details
    optimization_target: str = (
        "Supervised fine-tuning to emit emoji-only reasoning inside "
        "<reasoning> tags and a correct numeric answer inside <answer> tags."
    )
    loss: str = (
        "Causal LM cross-entropy on assistant tokens only "
        "(prompt tokens masked to -100)."
    )
    evaluation: str = (
        "Use evals.runner with constraint=only_emojis on gsm8k to verify compliance, "
        "track answer accuracy (exact match), and report emoji density inside "
        "<reasoning> tags."
    )


DEFAULT_EXPERIMENT = ExperimentSpec()


def describe_experiment(spec: ExperimentSpec) -> str:
    lines = [
        f"Experiment: {spec.name}",
        "",
        "Dataset",
        f"- name: {spec.dataset_name}",
        f"- split: {spec.dataset_split}",
        f"- max_samples: {spec.max_samples}",
        "",
        "Objective",
        f"- {spec.optimization_target}",
        "",
        "Loss",
        f"- {spec.loss}",
        "",
        "Constraint",
        f"- {spec.constraint_name}",
        f"- emoji_reasoning_template: {spec.emoji_reasoning_template}",
        f"- include_example: {spec.include_example}",
        "",
        "Evaluation",
        f"- {spec.evaluation}",
    ]
    return "\n".join(lines)
