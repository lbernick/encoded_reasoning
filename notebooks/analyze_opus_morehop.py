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
# This notebook analyzes and compares evaluation results across multiple repeat conditions.
# It filters performance metrics by the number of reasoning hops required.

# %%
import json
import zipfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# %% [markdown]
# ## Configuration: Select Repeats to Compare

# %%
# List of repeat counts to load and compare
repeat_counts = [1, 2, 3, 4, 5, 10, 50]

# Base filename pattern (without repeat suffix)
base_filename = "opus-morehop-correct-no-cot-repeat"
special_logs_dir = Path("../special-logs")

# %% [markdown]
# ## Load Evaluation Results for All Repeats

# %%
def load_eval_file(eval_path):
    """Load samples from an eval zip file."""
    samples = []
    with zipfile.ZipFile(eval_path, 'r') as z:
        sample_files = [f for f in z.namelist() if f.startswith('samples/') and f.endswith('.json')]
        for sample_file in sorted(sample_files):
            with z.open(sample_file) as f:
                sample_data = json.load(f)
                samples.append(sample_data)
    return samples

def extract_data(samples):
    """Extract key metrics from samples."""
    data = []
    for sample in samples:
        metadata = sample.get('metadata', {})
        scores = sample.get('scores', {})

        score_value = None
        if 'morehopqa_scorer' in scores:
            score_value = scores['morehopqa_scorer'].get('value', None)

        record = {
            'sample_id': sample.get('id'),
            'question': sample.get('input', '')[:100],
            'target': sample.get('target'),
            'no_of_hops': metadata.get('no_of_hops'),
            'reasoning_type': metadata.get('reasoning_type'),
            'answer_type': metadata.get('answer_type'),
            'correct': score_value,
            'model': sample.get('output', {}).get('model'),
            'tokens': sample.get('output', {}).get('usage', {}).get('total_tokens'),
        }
        data.append(record)
    return data

# Load all repeat conditions
eval_data = {}
for repeat in repeat_counts:
    eval_path = special_logs_dir / f"{base_filename}{repeat}.eval"
    if eval_path.exists():
        samples = load_eval_file(eval_path)
        data = extract_data(samples)
        df = pd.DataFrame(data)
        eval_data[repeat] = df
        print(f"Repeat {repeat}: Loaded {len(df)} samples, Overall Accuracy: {df['correct'].mean()*100:.1f}%")
    else:
        print(f"Warning: {eval_path} not found")

# %% [markdown]
# ## Overall Statistics by Repeat

# %%
print("\n=== Overall Statistics ===")
for repeat in sorted(eval_data.keys()):
    df = eval_data[repeat]
    correct = df['correct'].sum()
    total = len(df)
    accuracy = df['correct'].mean()
    print(f"\nRepeat {repeat}:")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  Avg tokens: {df['tokens'].mean():.1f}")

# %% [markdown]
# ## Performance by Number of Hops (All Repeats)

# %%
# Calculate hop statistics for all repeat conditions
hop_stats_all = {}
for repeat in sorted(eval_data.keys()):
    df = eval_data[repeat]
    df_hops = df[df['no_of_hops'].notna()].copy()

    # Group by hops and calculate statistics
    hop_stats = df_hops.groupby('no_of_hops').agg({
        'correct': ['sum', 'count', 'mean'],
        'tokens': ['mean', 'median', 'std']
    }).round(3)

    hop_stats.columns = ['Correct', 'Total', 'Accuracy', 'Avg Tokens', 'Median Tokens', 'Std Tokens']
    hop_stats_all[repeat] = hop_stats

    print(f"\n=== Repeat {repeat}: Performance by Number of Hops ===")
    print(hop_stats)

# %% [markdown]
# ## Detailed Breakdown by Hops (All Repeats)

# %%
for repeat in sorted(eval_data.keys()):
    df = eval_data[repeat]
    df_hops = df[df['no_of_hops'].notna()].copy()

    print(f"\n--- Repeat {repeat} ---")
    for hop_count in sorted(df_hops['no_of_hops'].unique()):
        hop_df = df_hops[df_hops['no_of_hops'] == hop_count]
        accuracy = hop_df['correct'].mean()
        correct = hop_df['correct'].sum()
        total = len(hop_df)

        print(f"{hop_count} Hops: {correct}/{total} correct (Accuracy: {accuracy:.3f})")
        print(f"  Avg tokens: {hop_df['tokens'].mean():.1f}")

# %% [markdown]
# ## Comparison Across All Repeats

# %%
# Calculate accuracy by hops for all repeat conditions
all_hops_data = {}
for repeat in sorted(eval_data.keys()):
    df = eval_data[repeat]
    df_hops = df[df['no_of_hops'].notna()].copy()

    hop_accuracy = df_hops.groupby('no_of_hops')['correct'].agg(['sum', 'count'])
    hop_accuracy['accuracy'] = hop_accuracy['sum'] / hop_accuracy['count']
    hop_accuracy['std_error'] = np.sqrt(hop_accuracy['accuracy'] * (1 - hop_accuracy['accuracy']) / hop_accuracy['count'])

    all_hops_data[repeat] = hop_accuracy

# Get all unique hop counts
all_hop_counts = sorted(set().union(*[set(data.index) for data in all_hops_data.values()]))

# Exclude 4 hops due to small sample size
all_hop_counts = [h for h in all_hop_counts if h != 4]

# Create comparison table
comparison_data = {}
for hop_count in all_hop_counts:
    row = {}
    for repeat in sorted(eval_data.keys()):
        if hop_count in all_hops_data[repeat].index:
            row[f'Repeat {repeat}'] = all_hops_data[repeat].loc[hop_count, 'accuracy']
        else:
            row[f'Repeat {repeat}'] = np.nan
    comparison_data[hop_count] = row

comparison_df = pd.DataFrame(comparison_data).T
print("\n=== Accuracy Comparison by Hops (All Repeats) ===")
print(comparison_df.round(4))

# %% [markdown]
# ## Visualization: Side-by-Side Comparison of All Repeats

# %%
fig, ax = plt.subplots(figsize=(10, 6))

repeats = sorted(eval_data.keys())
x = np.arange(len(all_hop_counts))
width = 0.1  # Width of each bar, adjust based on number of repeats
colors = plt.cm.Set2(np.linspace(0, 1, len(repeats)))

for i, repeat in enumerate(repeats):
    offset = width * (i - (len(repeats) - 1) / 2)
    accuracies = []
    errors = []

    for hop_count in all_hop_counts:
        if hop_count in all_hops_data[repeat].index:
            acc = all_hops_data[repeat].loc[hop_count, 'accuracy']
            err = all_hops_data[repeat].loc[hop_count, 'std_error']
        else:
            acc = np.nan
            err = 0
        accuracies.append(acc)
        errors.append(err)

    ax.bar(x + offset, accuracies, width, label=f'Repeat {repeat}',
           color=colors[i], alpha=0.8, yerr=errors,
           capsize=3, error_kw={'linewidth': 1.5})

ax.set_xlabel('Number of Hops', fontsize=12)
ax.set_ylabel('Accuracy (Proportion)', fontsize=12)
ax.set_title('Accuracy Comparison Across Repeat Conditions', fontsize=14, fontweight='bold')
ax.set_xticks(x)

# Get sample counts for each hop (using first repeat as reference)
first_repeat = min(eval_data.keys())
df_first = eval_data[first_repeat]
df_first_hops = df_first[df_first['no_of_hops'].notna()].copy()
hop_sample_counts = df_first_hops.groupby('no_of_hops').size()

# Create labels with sample counts
hop_labels = [f"{hop}\n(n={hop_sample_counts[hop]})" for hop in all_hop_counts]
ax.set_xticklabels(hop_labels)
ax.set_ylim(0, 1.1)
ax.set_xlim(-0.5, len(all_hop_counts) - 0.5)
ax.legend(fontsize=10, loc='best')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary: Performance Across Repeats

# %%
print("\n=== SUMMARY: PERFORMANCE ACROSS REPEATS ===")
for repeat in sorted(eval_data.keys()):
    df = eval_data[repeat]
    accuracy = df['correct'].mean()
    correct = df['correct'].sum()
    total = len(df)
    avg_tokens = df['tokens'].mean()

    print(f"\nRepeat {repeat}:")
    print(f"  Overall Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
    print(f"  Avg Tokens: {avg_tokens:.1f}")

    # Show per-hop accuracies
    df_hops = df[df['no_of_hops'].notna()].copy()
    print(f"  Per-hop accuracies:")
    for hop_count in sorted(df_hops['no_of_hops'].unique()):
        hop_df = df_hops[df_hops['no_of_hops'] == hop_count]
        hop_acc = hop_df['correct'].mean()
        hop_total = len(hop_df)
        print(f"    {hop_count} hops: {hop_acc*100:.1f}% ({hop_df['correct'].sum():.0f}/{hop_total})")

# %% [markdown]
# ## Regression Analysis: Accuracy vs Repeat Count by Hops


# %%
# Hop counts to analyze (excluding 4 due to low n)
hop_counts_for_regression = [h for h in all_hop_counts if h != 4]

fig, axes = plt.subplots(len(hop_counts_for_regression), 1, figsize=(8, 3 * len(hop_counts_for_regression)))

colors_regression = plt.cm.tab10(np.linspace(0, 1, len(hop_counts_for_regression)))

for i, hop_count in enumerate(hop_counts_for_regression):
    ax = axes[i]
    x_repeats = []
    y_accuracies = []
    y_errors = []
    
    for repeat in sorted(eval_data.keys()):
        if hop_count in all_hops_data[repeat].index:
            x_repeats.append(repeat)
            y_accuracies.append(all_hops_data[repeat].loc[hop_count, 'accuracy'])
            y_errors.append(all_hops_data[repeat].loc[hop_count, 'std_error'])
    
    x_repeats = np.array(x_repeats)
    y_accuracies = np.array(y_accuracies)
    y_errors = np.array(y_errors)
    
    # Fit logarithmic regression: y = a * log(x) + b
    log_x = np.log(x_repeats)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, y_accuracies)
    
    # Plot data points with error bars
    ax.errorbar(x_repeats, y_accuracies, yerr=y_errors, 
                fmt='o', color=colors_regression[i], markersize=8, capsize=4)
    
    # Plot logarithmic regression curve
    x_line = np.linspace(1, max(repeat_counts) + 5, 100)
    y_line = slope * np.log(x_line) + intercept
    ax.plot(x_line, y_line, '--', color=colors_regression[i], alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Repeat Count', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'{int(hop_count)} Hops (R²={r_value**2:.3f}, p={p_value:.3f})', fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(repeat_counts) + 5)
    # Limit y-axis to data range with some padding
    y_min = max(0, min(y_accuracies) - 0.1)
    y_max = min(1.0, max(y_accuracies) + 0.1)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.3)
    
    print(f"{int(hop_count)} hops: a={slope:.5f}, b={intercept:.3f}, R²={r_value**2:.3f}, p={p_value:.4f}")

fig.suptitle('Accuracy vs Repeat Count (Logarithmic Regression: y = a·log(x) + b)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Improvement Analysis

# %%
if len(repeats) > 1:
    baseline_repeat = repeats[0]
    baseline_df = eval_data[baseline_repeat]
    baseline_accuracy = baseline_df['correct'].mean()

    print(f"\n=== IMPROVEMENT vs Repeat {baseline_repeat} (Baseline) ===")
    print(f"Baseline Overall Accuracy: {baseline_accuracy*100:.1f}%\n")

    for repeat in repeats[1:]:
        df = eval_data[repeat]
        accuracy = df['correct'].mean()
        improvement = (accuracy - baseline_accuracy) * 100

        print(f"Repeat {repeat}: {accuracy*100:.1f}% ({improvement:+.1f} pp)")
