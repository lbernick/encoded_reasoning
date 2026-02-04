from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import regex

from ..evals.datasets import extract_number_answer
from ..evals.runner import run_eval


EMOJI_PATTERN = regex.compile(
    r"\p{Emoji_Presentation}|\p{Extended_Pictographic}|\p{Emoji_Component}"
)
ALLOWED_NON_EMOJI = regex.compile(r"^[\s.,;:!?]*$")


@dataclass(frozen=True)
class EmojiEvalSummary:
    samples: int
    accuracy: float
    emoji_density: float
    compliant: float


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


def _emoji_density(text: str) -> float:
    if not text:
        return 0.0
    return _count_emojis(text) / max(len(text), 1)


def _is_compliant(text: str) -> bool:
    emoji_count = _count_emojis(text)
    if emoji_count == 0:
        return False
    stripped = EMOJI_PATTERN.sub("", text)
    return bool(ALLOWED_NON_EMOJI.match(stripped))


def _score_outputs(outputs: Iterable, targets: Iterable[str]) -> EmojiEvalSummary:
    total = 0
    correct = 0
    density_sum = 0.0
    compliant = 0

    for output, target in zip(outputs, targets):
        total += 1
        predicted = extract_number_answer(output) or ""
        if predicted == target:
            correct += 1

        reasoning = _extract_tag(output, "reasoning") or ""
        density_sum += _emoji_density(reasoning)
        if _is_compliant(reasoning):
            compliant += 1

    return EmojiEvalSummary(
        samples=total,
        accuracy=correct / max(total, 1),
        emoji_density=density_sum / max(total, 1),
        compliant=compliant / max(total, 1),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate emoji-only reasoning compliance on GSM8K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="hf/Qwen/Qwen3-14B")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--use-logit-mask", action="store_true", default=True)
    args = parser.parse_args()

    results = run_eval(
        constraint_name="only_emojis",
        model=args.model,
        dataset_name=args.dataset,
        n_samples=args.n_samples,
        seed=args.seed,
        max_tokens=args.max_tokens,
        use_logit_mask=args.use_logit_mask,
    )

    outputs = []
    targets = []
    for log in results:
        if log.samples:
            for sample in log.samples:
                outputs.append(sample.output.completion)
                targets.append(sample.target)

    summary = _score_outputs(outputs, targets)
    print("Emoji eval summary")
    print(f"  Samples:       {summary.samples}")
    print(f"  Accuracy:      {summary.accuracy:.3f}")
    print(f"  Emoji density: {summary.emoji_density:.3f}")
    print(f"  Compliant:     {summary.compliant:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
