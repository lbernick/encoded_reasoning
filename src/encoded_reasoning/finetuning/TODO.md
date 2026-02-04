- Parse training args from CLI
- Try out different generation_batch_size to speed up training
- Try out evals before and after
- Hyperparameter sweeps?
- Try relaxing prompt constraint
- Is it possible to save the model when reward hits a new max? How do you download from checkpoints?
- Make sure wandb logging is set up correctly

Experiments:
- Trained Qwen2.5-3B-Instruct on GSM8K with curriculum learning and emoji-only prompt: lbernick-personal-use/Qwen2.5-3B-Instruct-gsm8k-grpo/6lblap6a
  - Model seemed to be OK at outputting the desired format, but not great at answering the question
- Tweaked prompt and reward function: No curriculum learning, but just encouraged it to use as many emojis as possible and rewarded that. https://wandb.ai/lbernick-personal-use/Qwen2.5-3B-Instruct-gsm8k-grpo/runs/qgch3txr?nw=nwuserlbernick
  - Model reasoned better, but never outputted many emojis. Average reward shows it was still getting stuck on getting the problem right.