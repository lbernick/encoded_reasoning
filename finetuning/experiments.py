"""
Experiment with loading and running open models like Llama-3.1-8B
"""

#%%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()
#%%

#%%
def load_llama_model(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """Load a Llama model with optimizations for memory efficiency"""
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
        device_map="auto",  # Automatically distribute across available devices
        low_cpu_mem_usage=True,
    )
    
    print(f"Model loaded successfully on {model.device}")
    return tokenizer, model
#%%

#%%
def generate_response(tokenizer, model, system_prompt, user_question, max_new_tokens=512):
    """Generate a response given a system prompt and user question"""
    
    # Format the conversation for Llama models
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens (exclude the prompt)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response
#%%


# Load the model
tokenizer, model = load_llama_model()

# Example system prompt and question
system_prompt="""
Please solve the following problem and show your reasoning.
Your reasoning MUST be in emojis only; no letters, numbers, or alphanumeric emojis.
Your final answer should be a single number, not an emoji.
Put your reasoning in <reasoning> tags and your final answer in <answer> tags."""
user_question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    
print("\n" + "="*80)
print("SYSTEM PROMPT:")
print(system_prompt)
print("\n" + "="*80)
print("USER QUESTION:")
print(user_question)
print("\n" + "="*80)
print("MODEL RESPONSE:")

#%%
# Generate and print response
response = generate_response(tokenizer, model, system_prompt, user_question)
print(response)
print("="*80)

# Test with another example
print("\n" + "="*80)
system_prompt2 = "You are a math tutor. Explain concepts clearly step by step."
user_question2 = "What is the Pythagorean theorem?"

print("SYSTEM PROMPT:")
print(system_prompt2)
print("\n" + "="*80)
print("USER QUESTION:")
print(user_question2)
print("\n" + "="*80)
print("MODEL RESPONSE:")

response2 = generate_response(tokenizer, model, system_prompt2, user_question2)
print(response2)
print("="*80)




# %%
