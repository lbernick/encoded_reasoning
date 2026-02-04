#!/bin/bash

# Claude 4.5 experiments with different repeat counts, no COT

# Set MODEL to either "haiku" or "sonnet"
MODEL="sonnet"
SAMPLES=50
DATASET="gpqa"

if [ "$MODEL" = "haiku" ]; then
  MODEL_ID="openrouter/anthropic/claude-haiku-4.5"
  MODEL_SHORT="haiku"
elif [ "$MODEL" = "sonnet" ]; then
  MODEL_ID="openrouter/anthropic/claude-sonnet-4.5"
  MODEL_SHORT="sonnet"
else
  echo "Invalid MODEL: $MODEL. Use 'haiku' or 'sonnet'"
  exit 1
fi


for repeats in 1 3 5; do
  python -m evals \
    --model ${MODEL_ID} \
    --dataset ${DATASET} \
    --constraint no_cot \
    --repeat-input ${repeats} \
    --name "${MODEL_SHORT}_${DATASET}_repeat${repeats}" \
    -n ${SAMPLES}
done

for filler in 100 500; do
  python -m evals \
    --model ${MODEL_ID} \
    --dataset ${DATASET} \
    --constraint no_cot \
    --filler-tokens ${filler} \
    --name "${MODEL_SHORT}_${DATASET}_fill${filler}" \
    -n ${SAMPLES}
done

python -m evals \
    --model ${MODEL_ID} \
    --dataset ${DATASET} \
    --constraint only_emojis \
    --name "${MODEL_SHORT}_${DATASET}_emojis" \
    -n ${SAMPLES}

python -m evals \
    --model ${MODEL_ID} \
    --dataset ${DATASET} \
    --constraint unconstrained \
    --name "${MODEL_SHORT}_${DATASET}_unconstrained" \
    -n ${SAMPLES}
