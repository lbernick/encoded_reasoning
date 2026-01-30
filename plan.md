## Goals

Do models have the capability to reason in ways that would be hard for humans to understand? If so, can we elicit this behavior?

Experiment #1: How many constraints can we put on models' reasoning before it starts to degrade performance?
- Ask models to reason and then answer. Constrain the reasoning via prompting.
- Ask open-weight models to reason and then answer. Constrain the reasoning via prompting, and ablate any logits that correspond to tokens we don't want to allow.

Experiment #2: Can we finetune a model to reason in an encoded way? This would be bad for safety though if we did it for real.
- RL-finetune a model to reason in a constrained way while still answering the question

Experiment #3: Can non-human-legible reasoning improve model performance if you include it in the prompt?
- Ask the model to reason about the problem in a constrained way without answering. Then, provide it with this reasoning and ask it to answer without doing additional reasoning.

## Findings
- gpt-4o-mini does do better on the gsm8k dataset when prompted to reason.
- gsm8k dataset maybe too easy for gpt-5-mini (but didn't test that hard)
- gpt-4o-mini mostly reasons in emojis if you ask it to, but this didn't help it get the right answer.

