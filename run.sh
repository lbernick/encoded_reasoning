#!/bin/bash
set -euo pipefail

# Load env vars from .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Install uv if not available
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Install project in editable mode
uv pip install -e .

# Run evals, forwarding any CLI args
uv run python -m encoded_reasoning "$@"
