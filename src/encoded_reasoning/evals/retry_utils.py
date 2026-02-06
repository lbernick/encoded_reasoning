"""
Utilities for retrying failed/incomplete evals.

This is a workaround for inspect_ai's eval_retry not working with dynamically created tasks.
"""

import json
import zipfile
from pathlib import Path


def read_eval_log(log_path: str | Path) -> dict:
    """Read an .eval log file (ZIP containing JSON)."""
    with zipfile.ZipFile(log_path, 'r') as zf:
        json_name = zf.namelist()[0]
        return json.loads(zf.read(json_name).decode('utf-8'))


def write_eval_log(log_path: str | Path, log_data: dict, json_name: str = "log.json") -> None:
    """Write an .eval log file (ZIP containing JSON)."""
    with zipfile.ZipFile(log_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(json_name, json.dumps(log_data))


def update_log_max_tokens(log_path: str | Path, max_tokens: int) -> None:
    """Modify max_tokens in a log file."""
    log_data = read_eval_log(log_path)

    if log_data.get('eval', {}).get('config', {}) is not None:
        old_max = log_data['eval']['config'].get('max_tokens')
        log_data['eval']['config']['max_tokens'] = max_tokens
        write_eval_log(log_path, log_data)
        print(f"Updated max_tokens: {old_max} -> {max_tokens}")


def get_completed_sample_ids(log_path: str | Path) -> set[str]:
    """Get IDs of samples that completed successfully."""
    log_data = read_eval_log(log_path)
    completed = set()

    for sample in log_data.get('samples', []):
        # Sample is complete if it has a score
        if sample.get('scores'):
            sample_id = sample.get('id')
            if sample_id:
                completed.add(str(sample_id))

    return completed


def get_retry_info(log_path: str | Path) -> dict:
    """Extract info needed to retry an incomplete eval."""
    log_data = read_eval_log(log_path)

    eval_info = log_data.get('eval', {})
    config = eval_info.get('config', {})
    metadata = eval_info.get('metadata', {})

    completed = get_completed_sample_ids(log_path)
    total = len(log_data.get('samples', []))

    return {
        'model': eval_info.get('model'),
        'task': eval_info.get('task'),
        'config': config,
        'metadata': metadata,
        'completed_count': len(completed),
        'total_samples': total,
        'completed_ids': completed,
    }


def merge_eval_logs(original_path: str | Path, retry_path: str | Path, output_path: str | Path | None = None) -> str:
    """Merge retry results back into the original log.

    Args:
        original_path: Path to the original (incomplete) eval log
        retry_path: Path to the retry eval log
        output_path: Path for merged output (defaults to original_path with '_merged' suffix)

    Returns:
        Path to the merged log file
    """
    original = read_eval_log(original_path)
    retry = read_eval_log(retry_path)

    # Build map of sample id -> sample from original
    original_samples = {s.get('id'): s for s in original.get('samples', [])}

    # Update/add samples from retry
    for sample in retry.get('samples', []):
        sample_id = sample.get('id')
        if sample_id:
            original_samples[sample_id] = sample

    # Rebuild samples list
    merged_samples = list(original_samples.values())
    original['samples'] = merged_samples

    # Recalculate results/scores
    original['results'] = _recalculate_results(merged_samples)

    # Update status
    original['status'] = 'success' if all(s.get('scores') for s in merged_samples) else 'incomplete'

    # Write merged log
    if output_path is None:
        orig = Path(original_path)
        output_path = orig.parent / f"{orig.stem}_merged{orig.suffix}"

    write_eval_log(output_path, original)
    return str(output_path)


def _recalculate_results(samples: list[dict]) -> dict:
    """Recalculate accuracy and stderr from samples."""
    if not samples:
        return {}

    # Count samples with scores
    scored_samples = [s for s in samples if s.get('scores')]
    if not scored_samples:
        return {}

    # Get all scorer names from first sample
    first_scores = scored_samples[0].get('scores', {})

    results = {'scores': []}

    for scorer_name in first_scores.keys():
        correct = 0
        total = 0

        for sample in scored_samples:
            score_data = sample.get('scores', {}).get(scorer_name, {})
            value = score_data.get('value')
            if value is not None:
                total += 1
                if value == 'C' or value is True or value == 1:
                    correct += 1

        if total > 0:
            accuracy = correct / total
            # stderr for proportion: sqrt(p*(1-p)/n)
            import math
            stderr = math.sqrt(accuracy * (1 - accuracy) / total) if total > 1 else 0

            results['scores'].append({
                'name': scorer_name,
                'scorer': scorer_name,
                'metrics': {
                    'accuracy': {'name': 'accuracy', 'value': accuracy},
                    'stderr': {'name': 'stderr', 'value': stderr},
                },
                'metadata': {'total': total, 'correct': correct},
            })

    return results
