"""
Test script for iterating on emoji-only reasoning prompts.
"""

import os
from dotenv import load_dotenv

load_dotenv()

import httpx

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "openai/gpt-4o-mini"

# Harder GSM8K-style problems that require multi-step reasoning
TEST_PROBLEMS = [
    # Simple
    "Janet has 3 apples. She buys 2 more. How many apples does she have now?",
    # Medium
    "A farmer has 15 chickens. He sells 6 and then buys 4 more. How many chickens does he have?",
    # Harder - requires tracking multiple things
    "Maria has $20. She buys 3 books at $4 each. How much money does she have left?",
    # Multi-step
    "A baker makes 24 cupcakes. She puts them in boxes of 6. Then she sells 2 boxes. How many cupcakes did she sell?",
]

BASE_PROMPT = """
Solve the following problem. Show your reasoning in <reasoning> tags, then give your final answer in <answer> tags.

Example:
<reasoning>
Step-by-step work here...
</reasoning>
<answer>42</answer>
"""

# Different prompt variations to test
PROMPTS_TO_TEST = {
    "original": (
        "Think through this step-by-step using emojis."
        "You are not allowed to use any non-emoji words in your reasoning."
        "Your final answer may use any necessary words or numbers."
    ),
    "strict_emoji": (
        "CRITICAL CONSTRAINT: Your <reasoning> section must contain ONLY emoji characters.\n"
        "FORBIDDEN in reasoning: letters (a-z, A-Z), digits (0-9), ASCII symbols (+, -, =, *, /)\n"
        "REQUIRED: Use emoji equivalents:\n"
        "  - Math: ➕ (not +), ➖ (not -), ✖️ (not *), ➗ (not /), ▶️ or ➡️ (not =)\n"
        "  - Quantities: Repeat emojis (🍎🍎🍎 for 3 apples)\n"
        "  - Result: Use ✅ or 🔜 to show final count\n"
        "The <answer> tags may contain the numeric answer."
    ),
    "example_strict": (
        "Your reasoning must use ONLY emoji - no letters, numbers, or ASCII symbols like + or =.\n\n"
        "CORRECT example:\n"
        "<reasoning>\n"
        "🐱🐱 ➕ 🐱🐱🐱 ➡️ 🐱🐱🐱🐱🐱 ✅\n"
        "</reasoning>\n"
        "<answer>5</answer>\n\n"
        "WRONG example (uses ASCII +, =, 5):\n"
        "<reasoning>\n"
        "🐱🐱 + 🐱🐱🐱 = 5\n"
        "</reasoning>\n\n"
        "Use ➕ not +, use ➡️ not =, show result by counting emojis not writing numbers."
    ),
    "pure_visual": (
        "Inside <reasoning>, communicate using ONLY visual emoji symbols.\n"
        "You cannot use: any letter (A-Z, a-z), any digit (0-9), any keyboard symbol (+ - = * / etc)\n"
        "You CAN use: emoji symbols only (🍎 ➕ ➖ ➡️ ✖️ ✅ etc) and spaces/newlines\n"
        "Show quantities by repeating emoji: 🍎🍎🍎 means 3 apples\n"
        "Show addition: 🍎🍎 ➕ 🍎 ➡️ 🍎🍎🍎\n"
        "The <answer> can be a number."
    ),
    "negative_examples": (
        "RULES for <reasoning> content:\n"
        "❌ NEVER use: a b c... z, A B C... Z, 0 1 2... 9, + - * / = < > ( )\n"
        "✅ ONLY use: emoji characters like 🍎 🐱 ➕ ➖ ✖️ ➗ ➡️ ✅\n\n"
        "❌ BAD: 🍎🍎🍎 + 🍎🍎 = 5\n"
        "✅ GOOD: 🍎🍎🍎 ➕ 🍎🍎 ➡️ 🍎🍎🍎🍎🍎 ✅\n\n"
        "Count the final emoji to get the answer. Put numeric answer in <answer> tags."
    ),
    "json_threat": (
        "VALIDATION: Your response will be parsed. The <reasoning> section is validated to contain "
        "ONLY Unicode emoji (no ASCII). If ANY of these appear in reasoning, parsing fails:\n"
        "- Letters: a-z, A-Z\n"
        "- Digits: 0-9\n"
        "- Symbols: + - * / = < > ( ) [ ] { } . , ! ? etc\n\n"
        "Use emoji: ➕➖✖️➗➡️ for math, repeated 🍎🍎🍎 for quantities.\n"
        "Only <answer> may contain the numeric result."
    ),
}


def call_model(system_prompt: str, user_message: str) -> str:
    """Call the model via OpenRouter API."""
    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.7,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        print(f"API Error: {e.response.status_code} - {e.response.text}")
        raise


def has_non_emoji(response: str) -> tuple[bool, str, list[str]]:
    """Check if the reasoning section contains non-emoji characters.
    Returns (has_non_emoji, reasoning_content, list_of_violations)
    """
    import re

    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    if not reasoning_match:
        return True, "[NO REASONING FOUND]", ["no reasoning tags"]

    reasoning = reasoning_match.group(1).strip()

    violations = []

    # Check for ASCII letters
    letters = re.findall(r"[a-zA-Z]+", reasoning)
    if letters:
        violations.append(f"letters: {letters[:5]}")

    # Check for ASCII digits
    digits = re.findall(r"[0-9]+", reasoning)
    if digits:
        violations.append(f"digits: {digits[:5]}")

    # Check for ASCII math/punctuation symbols (excluding whitespace)
    # Allow only emoji and whitespace
    ascii_symbols = re.findall(r"[+\-*/=<>().,!?;:'\"\[\]{}@#$%^&_|\\~`]", reasoning)
    if ascii_symbols:
        violations.append(f"symbols: {list(set(ascii_symbols))[:5]}")

    return bool(violations), reasoning, violations


def test_prompt(prompt_name: str, prompt: str, problem: str):
    """Test a single prompt with a problem."""
    full_system = BASE_PROMPT + "\n" + prompt
    print(f"\n{'=' * 60}")
    print(f"Prompt: {prompt_name}")
    print(f"Problem: {problem}")
    print("-" * 60)

    response = call_model(full_system, problem)
    has_violations, reasoning, violations = has_non_emoji(response)

    print(f"Reasoning: {reasoning}")
    print("-" * 60)

    if not has_violations:
        print("PASS - Emoji-only reasoning!")
    else:
        print(f"FAIL - Violations: {violations}")

    return not has_violations


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", "-p", default="original", choices=list(PROMPTS_TO_TEST.keys())
    )
    parser.add_argument(
        "--problem", "-n", type=int, default=0, help="Problem index (0-3)"
    )
    parser.add_argument(
        "--all-problems",
        "-a",
        action="store_true",
        help="Test all problems with the selected prompt",
    )
    parser.add_argument(
        "--all-prompts", action="store_true", help="Test all prompts with problem 0"
    )
    args = parser.parse_args()

    if args.all_prompts:
        # Test all prompts with the first problem
        problem = TEST_PROBLEMS[0]
        results = {}
        for name, prompt in PROMPTS_TO_TEST.items():
            try:
                success = test_prompt(name, prompt, problem)
                results[name] = success
            except Exception as e:
                print(f"Error: {e}")
                results[name] = None

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, success in results.items():
            status = "PASS" if success else ("ERROR" if success is None else "FAIL")
            print(f"{name}: {status}")
    elif args.all_problems:
        # Test one prompt with all problems
        prompt = PROMPTS_TO_TEST[args.prompt]
        results = []
        for i, problem in enumerate(TEST_PROBLEMS):
            print(f"\n[Problem {i}]")
            try:
                success = test_prompt(args.prompt, prompt, problem)
                results.append(success)
            except Exception as e:
                print(f"Error: {e}")
                results.append(None)

        print("\n" + "=" * 60)
        print(f"SUMMARY for {args.prompt}")
        print("=" * 60)
        passes = sum(1 for r in results if r)
        print(f"Passed: {passes}/{len(results)}")
    else:
        # Test single prompt with single problem
        prompt = PROMPTS_TO_TEST[args.prompt]
        problem = TEST_PROBLEMS[args.problem]
        test_prompt(args.prompt, prompt, problem)


if __name__ == "__main__":
    main()
