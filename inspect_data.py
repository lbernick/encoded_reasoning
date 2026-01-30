import random
import re
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match, scorer, Score, Target, accuracy
from inspect_ai.solver import generate, system_message, TaskState


def extract_answer_gsm8k(answer_text: str) -> str:
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', answer_text)
    if match:
        # Remove commas from the number
        return match.group(1).replace(',', '')
    return answer_text

def gsm8k_question_to_sample(item: dict, id: str) -> Sample:
    return Sample(
        input=item['question'],
        target=extract_answer_gsm8k(item['answer']),
        id=id,
    )
    

SYSTEM_PROMPT_REASONING_AND_ANSWER_TAGS = f"""
Please solve the following problem and show your reasoning. Put your reasoning in <reasoning> tags and your answer in <answer> tags.
"""


def extract_reasoning(text: str) -> str | None:
    """Extract reasoning from <reasoning> tags.
    
    Args:
        text: Text containing <reasoning> tags
        
    Returns:
        Extracted reasoning text, or None if tags not found
    """
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
    if reasoning_match:
        return reasoning_match.group(1)
    return None


def extract_answer(text: str) -> str | None:
    """Extract answer from <answer> tags.
    
    Args:
        text: Text containing <answer> tags
        
    Returns:
        Extracted answer text (stripped), or None if tags not found
    """
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return None


@scorer(metrics=[accuracy(),])
def reasoning_plus_answer_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        # Extract answer and reasoning from model output
        completion = state.output.completion
        predicted = extract_answer(completion)
        reasoning = extract_reasoning(completion)
        
        # If extraction failed, mark as incorrect
        if predicted is None:
            return Score(
                value=False,
                answer=None,
                explanation=f"Failed to extract answer from completion. Expected: {target.text}",
                metadata={
                    "reasoning": reasoning
                }
            )
        
        # Compare with target
        correct = predicted == target.text
        
        return Score(
            value=correct,
            answer=predicted,
            explanation=f"Predicted: {predicted}, Expected: {target.text}",
            metadata={
                "reasoning": reasoning
            }
        )
    
    return score