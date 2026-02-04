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
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    uv venv
fi

# Install project in editable mode
uv pip install -e .

# Run evals, forwarding any CLI args
uv run python -m encoded_reasoning "$@"
