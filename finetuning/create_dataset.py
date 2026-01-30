import random
import json

def generate_simple_math(n=500, system_prompt=""):
    problems = []
    for _ in range(n):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        op = random.choice(['+', '-', '*'])
        
        if op == '+':
            answer = a + b
        elif op == '-':
            # answer = a - b if a >= b else b - a
            # if a < b: a, b = b, a  # Keep positive
            answer = a - b
        elif op == "*":
            answer = a * b
        elif op == "/":
            answer = a / b
        else:
            answer = a ** b
        
        question = f"{system_prompt}What is {a} {op} {b}?" 
        problems.append({
            "question": question,
            "answer": str(answer)
        })
    return problems

if __name__ == "__main__":
    system_prompt = """
Please solve the following problem and show your reasoning.
Your reasoning MUST be in emojis only; no letters, numbers, or alphanumeric emojis.
Your final answer should be a single number, not an emoji.
Put your reasoning in <reasoning> tags and your final answer in <answer> tags.
"""
    problems = generate_simple_math(40, system_prompt)
    filename = "finetuning/data/simple_math_problems.json"
    with open(filename, "w") as f:
        json.dump(problems, f)
