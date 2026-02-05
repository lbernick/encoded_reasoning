#!/usr/bin/env python3
"""
Export inspect_ai .eval archives to a CSV summary.

Scans logs/*.eval (zip archives) and writes a CSV with columns:
Model, Dataset, Subset, Prompt, Constraint, Unfiltered prop correct,
Standard error, Prop. filtered out, eval file
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_OUT = PROJECT_ROOT / "eval_summary.csv"


def _read_json(zf: zipfile.ZipFile, name: str) -> Any | None:
    try:
        with zf.open(name) as f:
            return json.load(f)
    except Exception:
        return None


def _get_header(zf: zipfile.ZipFile) -> dict[str, Any] | None:
    return _read_json(zf, "header.json")


def _get_reductions(zf: zipfile.ZipFile) -> list[dict[str, Any]] | None:
    data = _read_json(zf, "reductions.json")
    if isinstance(data, list):
        return data
    return None


def _extract_accuracy(reductions: list[dict[str, Any]] | None) -> tuple[float | None, int]:
    if not reductions:
        return None, 0

    # Find first scorer with samples list of value 0/1
    for item in reductions:
        samples = item.get("samples")
        if not isinstance(samples, list) or not samples:
            continue
        values = [s.get("value") for s in samples]
        # keep numeric values only
        nums = [v for v in values if isinstance(v, (int, float))]
        if not nums:
            continue
        n = len(nums)
        acc = sum(nums) / n
        return acc, n

    return None, 0


def _stderr_from_acc(acc: float | None, n: int) -> float | None:
    if acc is None or n <= 1:
        return None
    # Standard error for Bernoulli proportion
    return (acc * (1 - acc) / n) ** 0.5


def _safe_get(d: dict[str, Any], path: list[str], default: str = "") -> str:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return str(cur)


def _parse_eval_file(eval_path: Path) -> dict[str, str]:
    row = {
        "Model": "",
        "Dataset": "",
        "Subset": "",
        "Prompt": "",
        "Constraint": "",
        "Unfiltered prop correct": "",
        "Standard error": "",
        "Prop. filtered out": "",
        "eval file": eval_path.name,
    }

    try:
        with zipfile.ZipFile(eval_path) as zf:
            header = _get_header(zf) or {}
            reductions = _get_reductions(zf)

            # Header fields
            row["Model"] = _safe_get(header, ["eval", "model"]) or _safe_get(
                header, ["eval", "model_name"]
            )
            row["Dataset"] = _safe_get(header, ["eval", "metadata", "dataset"]) or _safe_get(
                header, ["eval", "dataset", "name"]
            )
            # Subset: use configured limit if present, else sample count
            subset = _safe_get(header, ["eval", "config", "limit"])
            if subset:
                row["Subset"] = subset
            else:
                row["Subset"] = _safe_get(header, ["eval", "dataset", "samples"])

            row["Constraint"] = _safe_get(header, ["eval", "metadata", "constraint"])

            # Prompt intentionally left blank per export format

            # Metrics from reductions
            acc, n = _extract_accuracy(reductions)
            if acc is not None:
                row["Unfiltered prop correct"] = f"{acc:.3f}"
                stderr = _stderr_from_acc(acc, n)
                if stderr is not None:
                    row["Standard error"] = f"{stderr:.3f}"

    except zipfile.BadZipFile:
        # leave defaults, but still include filename
        pass

    return row


def main() -> None:
    eval_files = sorted(LOG_DIR.glob("*.eval"))
    out_path = DEFAULT_OUT

    fieldnames = [
        "Model",
        "Dataset",
        "Subset",
        "Prompt",
        "Constraint",
        "Unfiltered prop correct",
        "Standard error",
        "Prop. filtered out",
        "eval file",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for eval_path in eval_files:
            writer.writerow(_parse_eval_file(eval_path))

    print(f"Wrote {out_path} ({len(eval_files)} eval files)")


if __name__ == "__main__":
    main()
