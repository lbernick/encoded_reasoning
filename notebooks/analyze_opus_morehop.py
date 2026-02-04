# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # Analysis of Opus Model Performance on MoreHopQA
#
# This notebook analyzes the evaluation results from `opus-morehop-correct-no-cot.eval`,
# filtering performance metrics by the number of reasoning hops required.

# %%
import json
import zipfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Load the evaluation results

# %%
eval_path = Path("../special-logs/opus-morehop-correct-no-cot.eval")

# Extract and load all sample data
samples = []
with zipfile.ZipFile(eval_path, 'r') as z:
    # List all sample files
    sample_files = [f for f in z.namelist() if f.startswith('samples/') and f.endswith('.json')]

    for sample_file in sorted(sample_files):
        with z.open(sample_file) as f:
            sample_data = json.load(f)
            samples.append(sample_data)

print(f"Loaded {len(samples)} samples")
print(f"Sample keys: {samples[0].keys()}")

# %% [markdown]
# ## Extract and organize data

# %%
# Create a DataFrame with the key information
data = []
for sample in samples:
    metadata = sample.get('metadata', {})
    scores = sample.get('scores', {})

    # Get the score for the main scorer
    score_value = None
    if 'morehopqa_scorer' in scores:
        score_value = scores['morehopqa_scorer'].get('value', None)

    record = {
        'sample_id': sample.get('id'),
        'question': sample.get('input', '')[:100],  # First 100 chars
        'target': sample.get('target'),
        'no_of_hops': metadata.get('no_of_hops'),
        'reasoning_type': metadata.get('reasoning_type'),
        'answer_type': metadata.get('answer_type'),
        'correct': score_value,
        'model': sample.get('output', {}).get('model'),
        'tokens': sample.get('output', {}).get('usage', {}).get('total_tokens'),
    }
    data.append(record)

df = pd.DataFrame(data)
print(f"DataFrame shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# %% [markdown]
# ## Overall Statistics

# %%
print("=== Overall Statistics ===")
print(f"Total samples: {len(df)}")
print(f"Correct: {df['correct'].sum()} ({df['correct'].mean()*100:.1f}%)")
print(f"Incorrect: {(~df['correct']).sum()} ({(~df['correct']).mean()*100:.1f}%)")
print(f"Average tokens: {df['tokens'].mean():.1f}")
print(f"Median tokens: {df['tokens'].median():.1f}")

# %% [markdown]
# ## Total Token Count

# %%
# Extract input and output tokens
token_data = []
for sample in samples:
    usage = sample.get('output', {}).get('usage', {})
    token_data.append({
        'input_tokens': usage.get('input_tokens', 0),
        'output_tokens': usage.get('output_tokens', 0),
        'total_tokens': usage.get('total_tokens', 0),
    })

token_df = pd.DataFrame(token_data)

total_input_tokens = token_df['input_tokens'].sum()
total_output_tokens = token_df['output_tokens'].sum()
total_tokens = token_df['total_tokens'].sum()

print("=== Total Token Usage ===")
print(f"Total Input Tokens:  {total_input_tokens:,}")
print(f"Total Output Tokens: {total_output_tokens:,}")
print(f"Total Tokens:        {total_tokens:,}")
print(f"\nAverage per sample:")
print(f"  Input:  {token_df['input_tokens'].mean():.1f}")
print(f"  Output: {token_df['output_tokens'].mean():.1f}")
print(f"  Total:  {token_df['total_tokens'].mean():.1f}")

# %% [markdown]
# ## Performance by Number of Hops

# %%
# Filter out any rows with missing hop counts
df_hops = df[df['no_of_hops'].notna()].copy()

# Group by hops and calculate statistics
hop_stats = df_hops.groupby('no_of_hops').agg({
    'correct': ['sum', 'count', 'mean'],
    'tokens': ['mean', 'median', 'std']
}).round(3)

hop_stats.columns = ['Correct', 'Total', 'Accuracy', 'Avg Tokens', 'Median Tokens', 'Std Tokens']
print("=== Performance by Number of Hops ===")
print(hop_stats)

# %% [markdown]
# ## Detailed Breakdown by Hops

# %%
for hop_count in sorted(df_hops['no_of_hops'].unique()):
    hop_df = df_hops[df_hops['no_of_hops'] == hop_count]
    accuracy = hop_df['correct'].mean()
    correct = hop_df['correct'].sum()
    total = len(hop_df)

    print(f"\n{hop_count} Hops: {correct}/{total} correct (Accuracy: {accuracy:.3f})")
    print(f"  Avg tokens: {hop_df['tokens'].mean():.1f}")
    print(f"  Median tokens: {hop_df['tokens'].median():.1f}")
    print(f"  Min/Max tokens: {hop_df['tokens'].min():.0f}/{hop_df['tokens'].max():.0f}")

# %% [markdown]
# ## Visualization: Accuracy by Hops

# %%
fig, ax = plt.subplots(figsize=(10, 6))

# Accuracy by hops with standard error bars
hop_accuracy = df_hops.groupby('no_of_hops')['correct'].agg(['sum', 'count'])
hop_accuracy['accuracy'] = hop_accuracy['sum'] / hop_accuracy['count']
# Calculate standard error: sqrt(p(1-p)/n)
hop_accuracy['std_error'] = np.sqrt(hop_accuracy['accuracy'] * (1 - hop_accuracy['accuracy']) / hop_accuracy['count'])

x_pos = np.arange(len(hop_accuracy.index))
ax.bar(x_pos, hop_accuracy['accuracy'], yerr=hop_accuracy['std_error'],
            capsize=5, color='steelblue', alpha=0.7, error_kw={'linewidth': 2, 'ecolor': 'darkblue'})
ax.set_xlabel('Number of Hops', fontsize=12)
ax.set_ylabel('Accuracy (Proportion)', fontsize=12)
ax.set_title('Model Accuracy by Number of Reasoning Hops (with Standard Error)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(sorted(df_hops['no_of_hops'].unique()))
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (idx, row) in enumerate(hop_accuracy.iterrows()):
    ax.text(i, row['accuracy'] + row['std_error'] + 0.05, f"{row['sum']:.0f}/{row['count']:.0f}",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualization: Sample Distribution by Hops

# %%
fig, ax = plt.subplots(figsize=(10, 6))

hop_counts = df_hops.groupby('no_of_hops')['correct'].count()

x_pos = np.arange(len(hop_counts.index))
ax.bar(x_pos, hop_counts.values, color='coral', alpha=0.7)
ax.set_xlabel('Number of Hops', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Sample Distribution by Number of Hops', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(sorted(df_hops['no_of_hops'].unique()))
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (idx, count) in enumerate(hop_counts.items()):
    ax.text(i, count + 5, f"{count:.0f}",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualization: Token Usage by Hops

# %%
fig, ax = plt.subplots(figsize=(10, 6))

hop_tokens = df_hops.groupby('no_of_hops')['tokens'].agg(['mean', 'median', 'std'])

x = np.arange(len(hop_tokens.index))
width = 0.35

ax.bar(x - width/2, hop_tokens['mean'], width, label='Mean', color='steelblue', alpha=0.7)
ax.bar(x + width/2, hop_tokens['median'], width, label='Median', color='coral', alpha=0.7)

ax.set_xlabel('Number of Hops', fontsize=12)
ax.set_ylabel('Total Tokens', fontsize=12)
ax.set_title('Token Usage by Number of Hops', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sorted(df_hops['no_of_hops'].unique()))
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Error Analysis: Incorrect Predictions

# %%
incorrect = df_hops[~df_hops['correct']]
print(f"=== Incorrect Predictions ({len(incorrect)} total) ===\n")

for hop_count in sorted(incorrect['no_of_hops'].unique()):
    hop_incorrect = incorrect[incorrect['no_of_hops'] == hop_count]
    print(f"\n{hop_count} Hops ({len(hop_incorrect)} errors):")
    print(f"  Sample IDs: {sorted(hop_incorrect['sample_id'].tolist())[:10]}")
    if len(hop_incorrect) > 10:
        print(f"  ... and {len(hop_incorrect) - 10} more")

# %% [markdown]
# ## Summary

# %%
print("\n=== SUMMARY ===")
print(f"Model: {df['model'].iloc[0]}")
print(f"Dataset: MoreHopQA (Correct subset)")
print(f"Constraint: No Chain-of-Thought")
print(f"\nTotal Accuracy: {df['correct'].mean()*100:.1f}%")
print(f"\nAccuracy by Hops:")
for hop_count in sorted(df_hops['no_of_hops'].unique()):
    hop_df = df_hops[df_hops['no_of_hops'] == hop_count]
    acc = hop_df['correct'].mean()
    print(f"  {hop_count} hops: {acc*100:.1f}%")
