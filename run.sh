#!/bin/bash
set -euo pipefail

# Load env vars from .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Install project in editable mode
uv pip install -e .

# Run evals, forwarding any CLI args
python -m evals "$@"
