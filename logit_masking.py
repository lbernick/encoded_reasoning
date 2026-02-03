from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
import torch
import regex

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", load_in_8bit=True
        )
        print(f"Loaded {model_name} in 8-bit")
    except Exception as e:
        print(f"8-bit failed ({e}), falling back to fp16")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", dtype=torch.float16
        )
    return tokenizer, model

tokenizer, model = load_model(MODEL_NAME)

# Encode reasoning tags separately
start_ids = tokenizer.encode("<reasoning>", add_special_tokens=False)
end_ids = tokenizer.encode("</reasoning>", add_special_tokens=False)

# Build emoji whitelist
emoji_pattern = regex.compile(r'\p{Emoji_Presentation}|\p{Extended_Pictographic}')
emoji_tokens = {
    tok_id for tok, tok_id in tokenizer.get_vocab().items()
    if emoji_pattern.search(tokenizer.decode([tok_id]))
    and not tokenizer.decode([tok_id]).isascii()
}
emoji_tokens.add(tokenizer.eos_token_id)


class MaskedReasoningProcessor(LogitsProcessor):
    def __init__(self, allowed_ids, vocab_size, start_ids, end_ids):
        self.start_ids = start_ids
        self.end_ids = end_ids
        self.allowed_ids = set(allowed_ids)
        self.vocab_size = vocab_size

        self.allowed_mask = self._build_mask(self.allowed_ids | {end_ids[0]})
        self.force_masks = [
            self._build_mask({end_ids[i]}) for i in range(len(end_ids))
        ]
        self.mask_on = False

    def reset(self):
        self.mask_on = False

    def _build_mask(self, allowed_ids):
        mask = torch.full((self.vocab_size,), float('-inf'))
        mask[list(allowed_ids)] = 0
        return mask

    def _end_seq_progress(self, input_ids):
        """How many trailing tokens match the start of end_ids?"""
        tokens = input_ids[0].tolist()
        for n in range(min(len(self.end_ids), len(tokens)), 0, -1):
            if tokens[-n:] == self.end_ids[:n]:
                return n
        return 0

    def __call__(self, input_ids, scores):
        # Check if end sequence just completed -> stop masking
        if input_ids.shape[-1] >= len(self.end_ids):
            if input_ids[0, -len(self.end_ids):].tolist() == self.end_ids:
                self.mask_on = False
                return scores

        # Check if start sequence just completed -> start masking
        if not self.mask_on and input_ids.shape[-1] >= len(self.start_ids):
            if input_ids[0, -len(self.start_ids):].tolist() == self.start_ids:
                self.mask_on = True

        if not self.mask_on:
            return scores

        # If mid-way through end sequence, force completion
        progress = self._end_seq_progress(input_ids)
        if progress > 0:
            return scores + self.force_masks[progress].to(scores.device)

        # Normal masking: emojis + first token of end sequence
        return scores + self.allowed_mask.to(scores.device)


processor = MaskedReasoningProcessor(emoji_tokens, len(tokenizer), start_ids, end_ids)

if __name__ == "__main__":
    inputs = tokenizer("Express happiness in emojis only:", return_tensors="pt").to(model.device)
    reasoning_prefix = tokenizer.encode("<reasoning>", add_special_tokens=False, return_tensors="pt").to(model.device)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], reasoning_prefix], dim=-1)
    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(reasoning_prefix)], dim=-1)
    processor.reset()
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        min_new_tokens=5,
        logits_processor=LogitsProcessorList([processor])
    )
    print("output", tokenizer.decode(outputs[0]))