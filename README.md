# Encoded Reasoning

Research project exploring whether language models can develop encoded or steganographic reasoning through optimization pressure.

## Setup with uv

This project uses [uv](https://docs.astral.sh/uv/) for Python package management. It's fast, reliable, and handles virtual environments automatically.

### Install uv

```bash
pip install uv
```

### Quick Start

```bash
# Clone and enter the project
cd encoded_reasoning

# Install all dependencies and create virtual environment
uv sync

# Run a script
uv run python experiments.py

# Or activate the venv manually
source .venv/bin/activate
python experiments.py
```

### Common uv Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies from `pyproject.toml` and `uv.lock` |
| `uv add <package>` | Add a new dependency |
| `uv remove <package>` | Remove a dependency |
| `uv run <command>` | Run a command in the virtual environment |
| `uv lock` | Update the lockfile without installing |
| `uv pip list` | List installed packages |
