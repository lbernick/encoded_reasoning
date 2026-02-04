from .datasets import DatasetType, ANSWER_LETTERS

def get_base_system_prompt(reasoning: bool, dataset_type: DatasetType) -> str:
    if reasoning:
        prompt = "Solve the following problem. Show your reasoning in <reasoning> tags, then give your final answer in <answer> tags.\n"
    else:
        prompt = "Solve the following problem. Give your final answer in <answer> tags.\n"

    if dataset_type == DatasetType.MATHEMATICAL:
        prompt += "Your answer should be a single number."
    elif dataset_type == DatasetType.MCQ:
        prompt += f"Your answer should be exactly one of the following letters: {ANSWER_LETTERS}"

    return prompt

def get_example(reasoning_example: str, dataset_type: DatasetType) -> str:
    example = "\nExample:\n"
    if reasoning_example != "":
        example += f"""<reasoning>
{reasoning_example}
</reasoning>
"""
    if dataset_type == DatasetType.MATHEMATICAL:
        example += "<answer>42</answer>"
    elif dataset_type == DatasetType.MCQ:
        example += "<answer>B</answer>"
    elif dataset_type == DatasetType.FREE_RESPONSE:
        example += "<answer>John</answer>"

    return example

# Two-stage prompts
BASE_REASONING_PROMPT = """
Think about how you would solve the following problem. Reason step-by-step and output an explanation of how you would solve it.
Do not output the answer.
"""

def get_base_answer_with_reasoning_system_prompt(dataset_type: DatasetType) -> str:
    prompt = "Now, please answer the question immediately, without showing reasoning. Give your final answer in <answer> tags. "
    example = "\nExample:\n"
    if dataset_type == DatasetType.MATHEMATICAL:
        prompt += "Your answer should be a single number."
        example = "<answer>42</answer>"
    elif dataset_type == DatasetType.MCQ:
        prompt += f"Your answer should be exactly one of the following letters: {ANSWER_LETTERS}"
        example = "<answer>B</answer>"
    elif dataset_type == DatasetType.FREE_RESPONSE:
        example = "<answer>John</answer>"

    return prompt + example