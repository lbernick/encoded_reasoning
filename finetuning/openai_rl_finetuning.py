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
    reasoning_effort: str = "medium",
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
    grader_config = get_grader()
    
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


def upload_file(file_path: str, purpose: str = "fine-tune", expires_after_days: int = None):
    """
    Upload a file to OpenAI and return the file ID.
    
    Args:
        file_path: Path to the file to upload
        purpose: The intended purpose of the uploaded file. One of:
                 - "assistants": Used in the Assistants API
                 - "batch": Used in the Batch API
                 - "fine-tune": Used for fine-tuning (default)
                 - "vision": Images used for vision fine-tuning
                 - "user_data": Flexible file type for any purpose
                 - "evals": Used for eval data sets
        expires_after_days: Optional number of days until the file expires.
                           If None, files are persisted until manually deleted.
    
    Returns:
        str: The file ID (e.g., "file-abc123...")
    """
    with open(file_path, "rb") as f:
        if expires_after_days is not None:
            # Convert days to seconds
            expires_after_seconds = expires_after_days * 24 * 60 * 60
            file_obj = client.files.create(
                file=f,
                purpose=purpose,
                expires_after={
                    "anchor": "created_at",
                    "seconds": expires_after_seconds
                }
            )
        else:
            file_obj = client.files.create(
                file=f,
                purpose=purpose
            )
    
    print(f"✓ File uploaded successfully!")
    print(f"  File ID: {file_obj.id}")
    print(f"  Filename: {file_obj.filename}")
    print(f"  Bytes: {file_obj.bytes}")
    print(f"  Purpose: {file_obj.purpose}")
    if hasattr(file_obj, 'expires_at') and file_obj.expires_at:
        print(f"  Expires at: {file_obj.expires_at}")
    
    return file_obj.id


def main():
    #test_grader()
    #upload_file("finetuning/data/simple_math_problems_train.jsonl")
    #upload_file("finetuning/data/simple_math_problems_validation.jsonl")
    TRAINING_FILE_ID = "file-DVSHpMubLvVoYFbv48bjAN"
    VALIDATION_FILE_ID = "file-WJgD4MsJSgiiFy6JLC7fRa"
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
            model=MODEL,
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
