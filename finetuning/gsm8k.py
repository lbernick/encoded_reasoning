def extract_answer_from_gsm8k(answer_text: str) -> str:
    """
    Extract the final numerical answer from GSM8K format.
    GSM8K answers are in the format: "Step by step explanation\n#### 42"
    """
    # GSM8K format: answer is after ####
    if "####" in answer_text:
        answer = answer_text.split("####")[-1].strip()
        # Remove commas from numbers (e.g., "1,000" -> "1000")
        answer = answer.replace(",", "")
        return answer
    return answer_text.strip()