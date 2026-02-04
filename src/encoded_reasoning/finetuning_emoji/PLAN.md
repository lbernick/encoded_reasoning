# Emoji-Only Reasoning — Offline SDFT Plan (Option 1)

## Goal
Create an offline self-distilled SFT (SDFT) pipeline that teaches a HF local model to produce **emoji-only reasoning inside `<reasoning>` tags** while keeping **correct numeric answers** in `<answer>` tags. Avoid fixed emoji strings by using **teacher-generated emoji reasoning** as targets.

## High-Level Approach (Two-Pass SDFT)
1. **Pass 1: Teacher Reasoning Generation**
   - Use a teacher model (same as the student for this run).
   - Prompt with shared evals helpers (base prompt + dataset prompt).
   - Generate normal reasoning + answer:
     `<reasoning>...text reasoning...</reasoning><answer>...</answer>`
   - Extract the `<reasoning>` text for each sample.

2. **Pass 2: Emoji Translation (API + Prompting)**
   - Input: extracted `<reasoning>` text only (no question; reduce leakage).
   - Prompt: few-shot examples of rich emoji reasoning; instruction: “Express the reasoning as emojis that capture each step, preserve order; no letters/digits.”
   - Enforce emoji-only via **prompting + post-filter** (strip non-emoji); no logit masking.
   - Allow whitespace + basic punctuation; enforce non-empty emoji reasoning (>=1 emoji), cap ~256 tokens to curb rambling.

3. **Student SFT**
   - Build final SFT example§s:
     - `system`: base prompt + `only_emojis` constraint
     - `user`: GSM8K question
     - `assistant`: `<reasoning>{emoji_translation}</reasoning><answer>{gold_answer}</answer>`
   - Use causal LM loss on assistant tokens only (prompt masked).

4. **Evaluation**
   - Run `evals.runner` with `constraint=only_emojis` on GSM8K.
   - Add a reasoning compliance score (emoji-only + non-empty) for diagnostics.
   - Track answer accuracy (exact-match) and emoji-density inside `<reasoning>` tags.

---

## Dataset
- **Source:** `evals.datasets.load_dataset("gsm8k")`
- **Split:** `train`
- **Samples:** Configurable (start small, `n=10` for sanity checks; scale to 1,000)
- **Output Format (distilled):**
  ```
  <reasoning>...emoji-only...</reasoning>
  <answer>...</answer>
  ```

---

## Teacher Generation Details
- **Model:** configurable, local HF (same for teacher + student)
- **Samples:** 1,000 GSM8K questions (initial target)
- **Prompt:** base prompt + dataset prompt (no emoji constraint in Pass 1)
- **Artifacts:** JSONL with `{question, gold_answer, teacher_reasoning}` plus raw responses/prompts

## Emoji Translation Details
- **Model:** same as teacher (API generation)
- **Prompt:** few-shot emoji translations; translate reasoning → emoji-only reasoning
- **Decoding constraint:** none (prompt-only); optional post-filter to strip non-emoji
- **Artifacts:** JSONL with `{question, gold_answer, emoji_reasoning}` plus raw responses/prompts

---

## Student SFT Details
- **Model:** HF local
- **Loss:** Causal LM loss on assistant tokens (prompt masked to -100)
- **Trainer:** HF `Trainer` (or TRL `SFTTrainer` if desired)
- **LoRA/QLoRA:** optional

---

## Implementation Steps
1. **Add Distillation Script**
   - New file: `distill.py`
   - Pass 1: generate teacher reasoning for GSM8K.
   - Pass 2: translate reasoning → emoji-only reasoning.
   - Save final distilled JSONL for SFT.

2. **Modify Dataset Builder**
   - Load distilled JSONL when provided.
   - Use GSM8K only for distillation, not for direct training.

3. **Update Training Script**
   - SFT uses distilled dataset (`--distilled-path`).
   - Keep existing `train_sft.py` structure.

4. **Add Evaluation Hook**
   - Run `evals.runner` + reasoning compliance check.

---

## Success Criteria
- Reasoning is emoji-only (non-empty) for most outputs.
- Answer accuracy does not collapse vs baseline.
- No fixed reasoning string pattern dominates outputs.
