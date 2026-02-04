# %% [markdown]
# # MoreHopQA Dataset Analysis & Evaluation

# %%
# Setup path and imports
import sys

sys.path.insert(0, "..")

import pandas as pd
import json
import urllib.request
from pathlib import Path

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import system_message, generate, chain

from evals.datasets import morehopqa_scorer, get_dataset_system_prompt
from evals.runner import BASE_SYSTEM_PROMPT_COT

# %%
# Download and load the MoreHopQA dataset
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

morehopqa_url = "https://huggingface.co/datasets/alabnii/morehopqa/raw/main/data/with_human_verification.json"
filepath = data_dir / "morehopqa_with_human_verification.json"

if filepath.exists():
    print(f"Loading from {filepath}")
    with open(filepath, "r") as f:
        morehopqa_data = json.load(f)
else:
    print(f"Downloading from {morehopqa_url}")
    with urllib.request.urlopen(morehopqa_url) as response:
        morehopqa_data = json.loads(response.read().decode())
    with open(filepath, "w") as f:
        json.dump(morehopqa_data, f)
    print(f"Saved to {filepath}")

morehopqa_df = pd.DataFrame(morehopqa_data)
print(f"Loaded {len(morehopqa_df)} samples")

# %%
# Dataset overview
print(f"Shape: {morehopqa_df.shape}")
print(f"\nColumns: {morehopqa_df.columns.tolist()}")
morehopqa_df[["question", "answer", "no_of_hops", "reasoning_type"]].head(10)

# %%
# Distribution by number of hops
print("Distribution by no_of_hops:")
print(morehopqa_df["no_of_hops"].value_counts().sort_index())


# %%
# Stratified sampling: n samples from each no_of_hops class
def stratified_sample(df, column, n=50, seed=42):
    samples = []
    for value in sorted(df[column].unique()):
        group = df[df[column] == value]
        if len(group) <= n:
            samples.append(group)
        else:
            samples.append(group.sample(n=n, random_state=seed))
    return pd.concat(samples).reset_index(drop=True)


# Change n to adjust sample size per class
eval_subset = stratified_sample(morehopqa_df, "no_of_hops", n=50)
print(f"Eval subset size: {len(eval_subset)}")
print(f"\nSamples per no_of_hops:")
print(eval_subset["no_of_hops"].value_counts().sort_index())

# %%
# Build samples for evaluation
samples = [
    Sample(
        input=row["question"],
        target=str(row["answer"]),
        metadata={
            "question_id": row.get("_id", ""),
            "no_of_hops": row["no_of_hops"],
            "reasoning_type": row["reasoning_type"],
            "answer_type": row["answer_type"],
        },
    )
    for _, row in eval_subset.iterrows()
]

print(f"Built {len(samples)} samples for evaluation")

# %%
# Build the evaluation task
dataset_prompt = get_dataset_system_prompt("morehopqa") or ""
full_prompt = BASE_SYSTEM_PROMPT_COT + "\n" + dataset_prompt

task = Task(
    name="morehopqa_by_hops_sonnet",
    dataset=MemoryDataset(samples),
    solver=chain(
        system_message(full_prompt),
        generate(),
    ),
    scorer=morehopqa_scorer(),
)

print("Task created")
print(f"System prompt:\n{full_prompt[:300]}...")

# %%
# Run evaluation on Claude Sonnet
results = inspect_eval(
    task,
    model="openrouter/anthropic/claude-3.5-sonnet",
    log_dir="../logs",
)

# %%
# Display overall results
if results and len(results) > 0:
    r = results[0]
    if r.results and r.results.scores:
        score = r.results.scores[0]
        print("=== OVERALL RESULTS ===")
        print("Model: Claude 3.5 Sonnet")
        print(f"Samples: {len(r.samples) if r.samples else 'N/A'}")
        print(f"Accuracy: {score.metrics['accuracy'].value:.3f}")
        print(f"Stderr: {score.metrics['stderr'].value:.3f}")

# %%
# Break down results by number of hops
if results and len(results) > 0:
    r = results[0]

    # Build a dataframe of results
    results_data = []
    for sample in r.samples:
        no_of_hops = sample.metadata.get("no_of_hops", "unknown")
        score_value = sample.scores.get("morehopqa_scorer")
        is_correct = score_value.value if score_value else False
        results_data.append(
            {
                "no_of_hops": no_of_hops,
                "correct": is_correct,
                "predicted": score_value.answer if score_value else None,
                "expected": sample.target,
            }
        )

    results_df = pd.DataFrame(results_data)

    # Compute accuracy by no_of_hops
    print("\n=== RESULTS BY NUMBER OF HOPS ===")
    breakdown = results_df.groupby("no_of_hops").agg(
        correct=("correct", "sum"),
        total=("correct", "count"),
    )
    breakdown["accuracy"] = breakdown["correct"] / breakdown["total"]
    breakdown = breakdown.sort_index()
    print(breakdown.to_string())

    # Show some incorrect examples
    print("\n=== SAMPLE INCORRECT ANSWERS ===")
    incorrect = results_df[~results_df["correct"]].head(5)
    for _, row in incorrect.iterrows():
        print(f"\nHops: {row['no_of_hops']}")
        print(f"  Expected: {row['expected']}")
        print(f"  Predicted: {row['predicted']}")

# %%
# Token usage by number of hops
if results and len(results) > 0:
    r = results[0]

    # Extract token usage per sample
    token_data = []
    for sample in r.samples:
        no_of_hops = sample.metadata.get("no_of_hops", "unknown")

        # Get token usage from model output
        usage = sample.output.usage if sample.output else None
        if usage:
            token_data.append(
                {
                    "no_of_hops": no_of_hops,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                }
            )

    token_df = pd.DataFrame(token_data)

    print("=== TOKEN USAGE BY NUMBER OF HOPS ===")
    token_breakdown = (
        token_df.groupby("no_of_hops")
        .agg(
            avg_input=("input_tokens", "mean"),
            avg_output=("output_tokens", "mean"),
            avg_total=("total_tokens", "mean"),
            samples=("total_tokens", "count"),
        )
        .round(1)
    )
    token_breakdown = token_breakdown.sort_index()
    print(token_breakdown.to_string())

    print(f"\n=== OVERALL TOKEN USAGE ===")
    print(f"Avg input tokens:  {token_df['input_tokens'].mean():.1f}")
    print(f"Avg output tokens: {token_df['output_tokens'].mean():.1f}")
    print(f"Avg total tokens:  {token_df['total_tokens'].mean():.1f}")
    print(f"Total tokens used: {token_df['total_tokens'].sum()}")

    # Store for cost calculation
    est_input_tokens = token_df["input_tokens"].mean() * len(morehopqa_df)
    est_output_tokens = token_df["output_tokens"].mean() * len(morehopqa_df)

# %%
# Estimate total tokens for full dataset evaluation
if results and len(results) > 0:
    # Get average tokens per class from the eval
    avg_by_class = token_df.groupby("no_of_hops").agg(
        avg_input=("input_tokens", "mean"),
        avg_output=("output_tokens", "mean"),
    )

    # Get full dataset counts per class
    full_counts = morehopqa_df["no_of_hops"].value_counts().sort_index()

    print("=== ESTIMATED TOKENS FOR FULL DATASET ===\n")

    total_input = 0
    total_output = 0

    print(
        f"{'Hops':<6} {'Count':<8} {'Avg In':<10} {'Avg Out':<10} {'Est Input':<12} {'Est Output':<12}"
    )
    print("-" * 70)

    for hops in sorted(full_counts.index):
        count = full_counts[hops]
        if hops in avg_by_class.index:
            avg_in = avg_by_class.loc[hops, "avg_input"]
            avg_out = avg_by_class.loc[hops, "avg_output"]
            est_in = count * avg_in
            est_out = count * avg_out
            total_input += est_in
            total_output += est_out
            print(
                f"{hops:<6} {count:<8} {avg_in:<10.1f} {avg_out:<10.1f} {est_in:<12,.0f} {est_out:<12,.0f}"
            )

    print("-" * 70)
    print(
        f"{'TOTAL':<6} {full_counts.sum():<8} {'':<10} {'':<10} {total_input:<12,.0f} {total_output:<12,.0f}"
    )

    print(f"\n=== SUMMARY ===")
    print(f"Full dataset samples: {len(morehopqa_df)}")
    print(f"Estimated input tokens:  {total_input:,.0f}")
    print(f"Estimated output tokens: {total_output:,.0f}")
    print(f"Estimated total tokens:  {total_input + total_output:,.0f}")

    # Store for cost calculation
    est_input_tokens = total_input
    est_output_tokens = total_output

# %%
# Cost calculation
# Define your pricing here (cost per 1M tokens)
input_cost_per_million = 3.00  # e.g., $3.00 per 1M input tokens
output_cost_per_million = 15.00  # e.g., $15.00 per 1M output tokens
num_epochs = 1  # number of times to run each sample

# Calculate costs
input_cost = (est_input_tokens * num_epochs / 1_000_000) * input_cost_per_million
output_cost = (est_output_tokens * num_epochs / 1_000_000) * output_cost_per_million
total_cost = input_cost + output_cost

print("=== COST ESTIMATE ===")
print(f"Input token price:  ${input_cost_per_million:.2f} / 1M tokens")
print(f"Output token price: ${output_cost_per_million:.2f} / 1M tokens")
print(f"Epochs: {num_epochs}")
print()
print(f"Estimated input tokens:  {est_input_tokens * num_epochs:,.0f}")
print(f"Estimated output tokens: {est_output_tokens * num_epochs:,.0f}")
print()
print(f"Input cost:  ${input_cost:.2f}")
print(f"Output cost: ${output_cost:.2f}")
print(f"─────────────────────")
print(f"TOTAL COST: ${total_cost:.2f}")

# %%
# Run full evaluation and save correct answers
# WARNING: This will run on ALL 1118 samples - may take a while and cost money!

run_full_eval = False  # Set to True to run

if run_full_eval:
    # Build samples from FULL dataset
    full_samples = [
        Sample(
            input=row["question"],
            target=str(row["answer"]),
            metadata={
                "no_of_hops": row["no_of_hops"],
                "reasoning_type": row["reasoning_type"],
                "answer_type": row["answer_type"],
                "question_id": row.get("_id", idx),
            },
        )
        for idx, row in morehopqa_df.iterrows()
    ]

    print(f"Running full eval on {len(full_samples)} samples...")

    full_task = Task(
        name="morehopqa_full_sonnet",
        dataset=MemoryDataset(full_samples),
        solver=chain(
            system_message(full_prompt),
            generate(),
        ),
        scorer=morehopqa_scorer(),
    )

    full_results = inspect_eval(
        full_task,
        model="openrouter/anthropic/claude-3.5-sonnet",
        log_dir="../logs",
    )

    # Extract correct answers
    if full_results and len(full_results) > 0:
        fr = full_results[0]

        correct_questions = []
        incorrect_questions = []

        for sample in fr.samples:
            score_value = sample.scores.get("morehopqa_scorer")
            is_correct = score_value.value if score_value else False

            question_data = {
                "question": sample.input,
                "expected_answer": sample.target,
                "model_answer": score_value.answer if score_value else None,
                "no_of_hops": sample.metadata.get("no_of_hops"),
                "reasoning_type": sample.metadata.get("reasoning_type"),
                "answer_type": sample.metadata.get("answer_type"),
            }

            if is_correct:
                correct_questions.append(question_data)
            else:
                incorrect_questions.append(question_data)

        # Save to files
        correct_path = data_dir / "correct_answers.json"
        incorrect_path = data_dir / "incorrect_answers.json"

        with open(correct_path, "w") as f:
            json.dump(correct_questions, f, indent=2)

        with open(incorrect_path, "w") as f:
            json.dump(incorrect_questions, f, indent=2)

        print(f"\n=== FULL EVAL RESULTS ===")
        print(f"Total samples: {len(fr.samples)}")
        print(
            f"Correct: {len(correct_questions)} ({len(correct_questions) / len(fr.samples) * 100:.1f}%)"
        )
        print(
            f"Incorrect: {len(incorrect_questions)} ({len(incorrect_questions) / len(fr.samples) * 100:.1f}%)"
        )
        print(f"\nSaved correct answers to: {correct_path}")
        print(f"Saved incorrect answers to: {incorrect_path}")
else:
    print("Set run_full_eval = True to run the full evaluation")

# %%
