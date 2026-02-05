"""
Main entry point for encoded-reasoning package.

This delegates to the evals module's CLI.
"""

try:
    from .evals.__main__ import main
except ImportError:
    from encoded_reasoning.evals.__main__ import main

if __name__ == "__main__":
    main()
