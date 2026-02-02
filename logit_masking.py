from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
import torch
import regex

#model_name = "Qwen/Qwen2.5-72B-Instruct",
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Try 8-bit first, fall back to fp16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True
        )
        print(f"Loaded {model_name} in 8-bit")
    except Exception as e:
        print(f"8-bit failed ({e}), falling back to fp16")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.float16
        )
    
    return tokenizer, model

tokenizer, model = load_model(MODEL_NAME)

# Build emoji whitelist
reasoning_tags = ["<reasoning>", "</reasoning>"]
emoji_pattern = regex.compile(r'\p{Emoji_Presentation}|\p{Extended_Pictographic}')
emoji_tokens = {
    tok_id for tok, tok_id in tokenizer.get_vocab().items()
    if emoji_pattern.search(tokenizer.decode([tok_id])) 
    and not tokenizer.decode([tok_id]).isascii()
}
end_of_reasoning_tokens = (
    tokenizer.encode(reasoning_tags[-1], return_tensors="pt", add_special_tokens=False)
    .to(device)
)
reasoning_tokens = tokenizer.encode(reasoning_tags[0], add_special_tokens=False) + end_of_reasoning_tokens.squeeze(0).tolist()
emoji_tokens.add(tokenizer.eos_token_id)
emoji_tokens.update(reasoning_tokens)

class EmojiOnlyProcessor(LogitsProcessor):
    def __init__(self, allowed_ids, vocab_size, end_sequence_ids):
        self.mask = torch.full((vocab_size,), float('-inf'))
        self.mask[list(allowed_ids)] = 0
        self.end_sequence_ids = end_sequence_ids
        self.mask_on = True
    
    def reset(self):
        self.mask_on = True
    
    def __call__(self, input_ids, scores):
        seq_len = self.end_sequence_ids.shape[-1]
        if input_ids.shape[-1] >= seq_len:
            if torch.equal(input_ids[:, -seq_len:], self.end_sequence_ids):
                self.mask_on = False
        
        if self.mask_on:
            return scores + self.mask.to(scores.device)
        return scores

processor = EmojiOnlyProcessor(emoji_tokens, len(tokenizer), end_of_reasoning_tokens)

if __name__ == "__main__":
    # Generate
    inputs = tokenizer("Express happiness in emojis only:", return_tensors="pt").to(model.device)
    processor.reset()

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        min_new_tokens=5,  
        logits_processor=LogitsProcessorList([processor])
    )
    print("output", tokenizer.decode(outputs[0]))