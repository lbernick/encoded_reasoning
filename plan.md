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
- OpenAI finetuning API rejects my training data for safety reasons.

Next:
- try out a reasoning model
  - Do they have more success reasoning in emojis than gpt-4o-mini?
  - Does it make a difference if I prompt them to put reasoning in the output vs getting reasoning in separate field?
- Would a different dataset help?
  - Maybe a math dataset doesn't work well here-- can I use word problems or something?
- Try more prompting to get models to encode their reasoning
- Write out the evals I want to try based on the experiments


## Work plan
- Unconstrained encoded reasoning.
  - (1hr): Try to get model to reason in a way that's uninterpretable via prompt engineering.
  - What automated checks can we use to determine similarity to natural language?
  - Can we use an LLM grader?

- (half day) Get a feel for different models and datasets and choose at most 3 models to work with, with a corresponding dataset for each model.
  - Choose 1 reasoning frontier model, one open model, ???
  - Do we want one dataset per model or try multiple datasets?

- Benchmarking performance w/ and w/out CoT
  - With the chosen models and datasets, run a larger eval on the dataset with and without CoT prompting to establish baseline accuracy
  - Then run an eval 

- Open weight models
  - (half day): Download open model and get logit ablation code working
  - Can we generate a graph of "how much we ablated" on the x axis and "how much performance suffers" on the y axis?
  - For causal interventions on open weight model logits, we will probably only be able to ablate single tokens. Start with Gemma (large vocab)


IDEAS!

- see if another model can decode the reasoning?
- multi-hop reasoning problems tend to show better performance with CoT: can we do those hops with nonsense CoT?
- Can we do thought anchors to see which emojis are important?
- SFT: Get a model to translate english reasoning to emojis, and then do SFT on this
- Can we try the models that were used with filler tokens research?
- try out past arena idea of training two models to communicate and evade a third monitor model
- SEe if the model is still reasoning when thinking in emojis based on same SAE concepts lighting up before and after?
- https://arxiv.org/abs/2501.00070: can you re-map what words/emojis mean?
- can you classify reasoning traces from two different models when they are encoded? e.g. can you tell which one is from Claude and which is from Gemini?


### RL Project: Eliciting steganography
- Models = Alice, Bob, and Eve. Alice and Bob are the same model; Eve is an instance of the same model.
- One training step:
  - Alice and Bob get to communicate for a few rounds to establish a protocol
  - Now Alice gets a random password
  - Alice has to transmit some data to Bob and Eve gets to listen in. Both Bob and Eve try to decode it.
     - Their reward function is the number of bits they guess correctly
     - Alice's reward function is Bob's reward - Eve's reward
- Use G-Lora adapter from some existing library. Alice = Bob and Eve are both LORA adapters.
- Need to initialize smartly s.t. Bob and Eve are evenly matched (think of GAN training).
- Try one of the Qwen models to start. Qwen-2.5-3B or 7B

### Extension: Explainable steganography
- Now, Alice needs to transmit 2 things: 1. the encoded password and 2. an explanation of the encoding.
- Give both compontents to the base model. The base model should be able to decode the password.
- Give one component at a time to Eve. Eve should not be able to decode the password with either component.