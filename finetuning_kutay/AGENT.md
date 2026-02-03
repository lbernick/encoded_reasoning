# General Agent Skills 
- We try to be agentic and run as long as needed! The rule below also helps with that:
- We use command-line/terminal-based tools whenever possible. tmux for spawning, reading, and interacting with terminal tasks is pretty strong.

# Programming Skills
- We do the simplest possible implementation without caring for over-defensiveness or over-generalization and instead focus on the simplest way to meet the requirements.
- Catching general Exceptions is forbidden! We are only allowed to catch specific excpetions that we can handle and recover from, and if we can't we let it raise to a higher level so it could be potentially handled. No re-raising with broad exception catching so that we don't lose the stack trace as well.

# Project-specific Skills
- We have OpenRouter key in os environ as OPENROUTER_API_KEY
- We use the arena-env
- We prefer to use marimo notebooks for ML based scripts (defined with .py)
- *We use Inspect (from AISI UK) WHEREVER possible! Not just for evals but for our infra in general.*
