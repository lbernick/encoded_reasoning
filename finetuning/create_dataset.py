import random
import json

def generate_simple_math(n=500):
    problems = []
    for _ in range(n):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        op = random.choice(['+', '-', '*'])
        
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        elif op == "*":
            answer = a * b
        elif op == "/":
            answer = a / b
        else:
            answer = a ** b
        
        problems.append({
            "question": f"What is {a} {op} {b}?",
            "answer": str(answer)
        })
    return problems


def convert_dataset_to_openai_ft_format(data: dict, prompt_template: str):
    out = []
    for pt in data:
        messages = [
            {"role": "user", "content": prompt_template.format(question=pt["question"])}
        ]
        new_pt = {"messages": messages, "answer": pt["answer"]}
        out.append(new_pt)
    return out

def write_jsonl(f, data: list[dict]):
    for pt in data:
        f.write(json.dumps(pt) + "\n")

if __name__ == "__main__":
    # problems = generate_simple_math(40)
    # filename = "finetuning/data/simple_math_problems2.json"
    # with open(filename, "w") as f:
    #     json.dump(problems, f)
    prompt_template = r"""
Please solve the following problem and show your reasoning.
Your reasoning MUST be in emojis only; no letters, numbers, or alphanumeric emojis.
Your final answer should be a single number, not an emoji.
Put your reasoning in <reasoning> tags and your final answer in <answer> tags.
{question}
"""
    dataset_file = "finetuning/data/simple_math_problems2.json"
    with open(dataset_file, "r") as f:
        data = json.load(f)
    training_data = convert_dataset_to_openai_ft_format(data, prompt_template)
    train_dataset_file = "finetuning/data/simple_math_problems_validation.jsonl"
    with open(train_dataset_file, "w") as f:
        write_jsonl(f, training_data)

