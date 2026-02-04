"""
Constrained reasoning evaluation framework.

Usage:
    python -m evals -c baseline -n 10
    python -m evals -c no_cot -m openrouter/anthropic/claude-3.5-sonnet -n 50
"""

from .datasets import load_dataset, get_scorer, DATASETS
from .constraints import get_constraint, CONSTRAINTS, ReasoningConstraint
from .runner import run_eval, LOG_DIR
