- Add functionality for upload to huggingface
- Parse training args from CLI
- Keep a record of findings (link to wandb runs)
- Try out different generation_batch_size to speed up training
- Try out evals before and after
- Hyperparameter sweeps?
- Try relaxing prompt constraint
- Is it possible to save the model when reward hits a new max? How do you download from checkpoints?
- Make sure wandb logging is set up correctly

Experiments:
- Trained Qwen2.5-3B-Instruct on GSM8K with curriculum learning and emoji-only prompt: lbernick-personal-use/Qwen2.5-3B-Instruct-gsm8k-grpo/6lblap6a