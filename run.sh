#!/bin/bash

source /path/to/your/virtualenv/bin/activate


MODEL_NAME="google/flan-t5-base"
BATCH_SIZE=8
LEARNING_RATE=0.0001
TOTAL_EPOCHS=5

export NUM_TASKS=2
export PROMPT_TUNING_INIT_TEXT="edit text from source to target:"

python train.py --model_name $MODEL_NAME --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --total_epochs $TOTAL_EPOCHS
