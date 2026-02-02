"""
Emoji Reasoning RL Training
============================

RL training on GSM8K with a custom reward function that requires
emoji-only reasoning and correct numerical answers.

Based on tinker_cookbook/recipes/rl_loop.py
"""

import logging
import time
from concurrent.futures import Future

import chz
import datasets
import tinker
import torch
from tinker import types
from tqdm import tqdm
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from dotenv import load_dotenv
from grader import grade_output

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)
load_dotenv()


@chz.chz
class Config:
    """Training configuration."""
    base_url: str | None = None
    log_path: str = "./logs/emoji_rl"
    model_name: str = "meta-llama/Llama-3.1-8B"
    batch_size: int = 16
    group_size: int = 16
    learning_rate: float = 4e-5
    lora_rank: int = 32
    save_every: int = 20
    max_tokens: int = 512
    ttl_seconds: int = 604800  # 7 days
    wandb_project: str | None = "emoji-finetuning"
    wandb_name: str | None = None
    num_batches: int | None = 2  # If set, limits training to this many batches

SYSTEM_PROMPT="""
Please solve the following problem and show your reasoning.
Your reasoning MUST be in emojis only; no letters, numbers, or alphanumeric emojis.
Your final answer should be a single number, not an emoji.
Put your reasoning in <reasoning> tags and your final answer in <answer> tags."""


def get_reward(response: str, answer: str) -> float:
    """Calculate reward using the grade_output function."""

    # TODO: Make sure I'm actually pulling the asnwer out of the dataset correctly
    return grade_output(response, answer)


def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load GSM8K dataset
    logger.info("Loading dataset...")
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"]

    max_possible_batches = len(train_dataset) // config.batch_size
    n_train_batches = config.num_batches if config.num_batches is not None else max_possible_batches
    n_train_batches = min(n_train_batches, max_possible_batches)  # Don't exceed dataset size

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    # resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    # if resume_info:
    #     training_client = service_client.create_training_client_from_state_with_optimizer(
    #         resume_info["state_path"]
    #     )
    #     start_batch = resume_info["batch"]
    #     logger.info(f"Resuming from batch {start_batch}")
    # else:
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )
    start_batch = 0

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    
    # Optimizer params
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(f"Training for {n_train_batches} batches")

    # Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
                ttl_seconds=config.ttl_seconds,
            )

        # Get training batch
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        # Save weights for sampling
        sampling_path = (
            training_client.save_weights_for_sampler(
                name=f"{batch_idx:06d}", ttl_seconds=config.ttl_seconds
            )
            .result()
            .path
        )
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        futures_P: list[Future[types.SampleResponse]] = []
        prompts_P: list[types.ModelInput] = []
        
        #breakpoint()
        # Create prompts with system prompt
        for question in batch_rows["question"]:
            convo = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            model_input = renderer.build_generation_prompt(convo)

            # Generate group_size responses in a single call
            future = sampling_client.sample(
                prompt=model_input,
                num_samples=config.group_size,
                sampling_params=sampling_params,
            )
            futures_P.append(future)
            prompts_P.append(model_input)

        # Process responses
        for future, prompt, answer in tqdm(
            zip(futures_P, prompts_P, batch_rows["answer"]),
            total=len(futures_P),
            desc=f"Sampling batch {batch_idx}",
        ):
            sample_result = future.result()
            rewards_G: list[float] = []
            sampled_tokens_G_T: list[list[int]] = []
            logprobs_G_T: list[list[float]] = []
            
            for sequence in sample_result.sequences:
                sampled_tokens = sequence.tokens
                sampled_logprobs = sequence.logprobs
                assert sampled_logprobs is not None

                sampled_tokens_G_T.append(sampled_tokens)
                logprobs_G_T.append(sampled_logprobs)
        
                parsed_message, _ = renderer.parse_response(sampled_tokens)
                content = renderers.get_text_content(parsed_message)
                parsed_answer = extract_gsm8k_final_answer(answer)
                reward = get_reward(content, parsed_answer)
                rewards_G.append(reward)

            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [reward - mean_reward for reward in rewards_G]
            rewards_P.append(mean_reward)

            # Skip if all advantages are zero
            if all(advantage == 0.0 for advantage in advantages_G):
                continue

            # Create datums for training
            for sampled_tokens, logprobs, advantage in zip(
                sampled_tokens_G_T, logprobs_G_T, advantages_G
            ):
                ob_len = prompt.length - 1
                model_input = prompt.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))
                target_tokens = [0] * ob_len + sampled_tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                
                assert (
                    model_input.length
                    == len(target_tokens)
                    == len(padded_logprobs)
                    == len(padded_advantages)
                )
                
                datum = types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                )
                datums_D.append(datum)

        # Training step
        fwd_bwd_future = training_client.forward_backward(datums_D, loss_fn="importance_sampling")
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_step_future.result()

        if optim_result.metrics:
            metrics.update(optim_result.metrics)

        # Log metrics
        metrics["time/total"] = time.time() - t_start
        metrics["reward/total"] = sum(rewards_P) / len(rewards_P) if rewards_P else 0.0
        ml_logger.log_metrics(metrics, step=batch_idx)
        
        logger.info(f"Batch {batch_idx}: reward={metrics['reward/total']:.3f}, time={metrics['time/total']:.1f}s")

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
        ttl_seconds=config.ttl_seconds,
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)