from .dataset import DatasetBuildConfig, build_hf_sft_dataset
from .experiment import DEFAULT_EXPERIMENT

cfg = DatasetBuildConfig(
    dataset_name=DEFAULT_EXPERIMENT.dataset_name,
    split=DEFAULT_EXPERIMENT.dataset_split,
    max_samples=20,
    constraint_name=DEFAULT_EXPERIMENT.constraint_name,
    include_example=DEFAULT_EXPERIMENT.include_example,
    emoji_reasoning_template=DEFAULT_EXPERIMENT.emoji_reasoning_template,
)

print("Building dataset...")

try:
    ds = build_hf_sft_dataset(cfg)
except Exception as e:
    print("ERROR building dataset:", type(e).__name__, e)
    raise

print("Dataset rows:", len(ds))
for i in range(len(ds)):
    row = ds[i]
    print("\\n=== ROW", i, "===")
    print("Question:", row["question"])
    print("Answer:", row["answer"])
    print("Messages:")
    for msg in row["messages"]:
        preview = msg["content"].replace("\\n", "\\\\n")
        print("-", msg["role"], ":", preview[:220])