#!/usr/bin/env python3
"""
Script to submit a reinforcement fine-tuning job to OpenAI API.

This script creates an RFT job using a Python grader defined in grader.py.
The grader validates that model outputs contain emoji-only reasoning and correct answers.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import importlib
import inspect

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
assert API_KEY is not None
client = OpenAI(api_key=API_KEY)

def get_grader():
    grader_module = importlib.import_module("grader")
    return {
        "type": "python",
        "source": inspect.getsource(grader_module)
    }


def create_rft_job(
    training_file_id: str,
    validation_file_id: str,
    model: str = "o4-mini-2025-04-16",
    reasoning_effort: str = "medium"
):
    """
    Create a reinforcement fine-tuning job with a Python grader.
    
    Args:
        training_file_id: ID of the uploaded training file (e.g., "file-abc123...")
        validation_file_id: ID of the uploaded validation file (e.g., "file-xyz789...")
        model: Base model to fine-tune (default: "o4-mini-2025-04-16")
        reasoning_effort: Reasoning effort level - "low", "medium", or "high" (default: "medium")
    
    Returns:
        dict: The fine-tuning job object
    """
    grader_code = get_grader()
    
    # Configure the Python grader
    # The grader function takes 'text' (model output) and 'correct_answer' as parameters
    grader_config = {
        "type": "python",
        "code": grader_code,
        "input": {
            "text": "{{sample.output}}",
            "correct_answer": "{{item.correct_answer}}"
        }
    }
    
    # Create the fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model,
        method={
            "type": "reinforcement",
            "reinforcement": {
                "grader": grader_config,
                "hyperparameters": {
                    "reasoning_effort": reasoning_effort
                }
            }
        }
    )
    
    return job

def test_grader():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    grader = get_grader()

    # validate the grader
    payload = {"grader": grader}
    response = requests.post(
        "https://api.openai.com/v1/fine_tuning/alpha/graders/validate",
        json=payload,
        headers=headers
    )
    print("validate request_id:", response.headers["x-request-id"])
    print("validate response:", response.text)

    # run the grader with a test reference and sample
    payload = {
    "grader": grader,
    "item": {
        "answer": "45"
    },
    "model_sample": "<reasoning>🌼🌸 ➡️💜💜💜</reasoning><answer>45</answer>"
    }
    response = requests.post(
        "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
        json=payload,
        headers=headers
    )
    print("run request_id:", response.headers["x-request-id"])
    print("run response:", response.text)



def main():
    TRAINING_FILE_ID = "finetuning/data/simple_math_problems.json"
    VALIDATION_FILE_ID = "finetuning/data/simple_math_problems.json"
    MODEL = "o4-mini-2025-04-16"
    
    print("Creating reinforcement fine-tuning job...")
    print(f"  Training file: {TRAINING_FILE_ID}")
    print(f"  Validation file: {VALIDATION_FILE_ID}")
    print(f"  Base model: {MODEL}")
    print()
    
    try:
        job = create_rft_job(
            training_file_id=TRAINING_FILE_ID,
            validation_file_id=VALIDATION_FILE_ID,
            model=MODEL
        )
        
        print("✓ Fine-tuning job created successfully!")
        print(f"  Job ID: {job.id}")
        print(f"  Status: {job.status}")
        print(f"  Created at: {job.created_at}")
        print()
        print(f"Monitor your job at: https://platform.openai.com/finetune/{job.id}")
        
    except Exception as e:
        print(f"✗ Error creating fine-tuning job: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
