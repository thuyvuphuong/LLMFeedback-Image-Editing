#!/bin/bash

#SBATCH --job-name=llm_pretrained_unet_1000steps_4nextepochs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx-small
#SBATCH --output=train_outs/LLMfeedback/out/%x.%j.out
#SBATCH --error=train_outs/LLMfeedback/errors/%x.%j.err
#SBATCH --mail-type=ALL

# Model and dataset paths
export MODEL_NAME="pretrained_frameworks/pretrained_IEDMs/instruct-pix2pix"
export UNET_MODEL_PATH="finetuned_models/ip2p_nollm_res256_lr5e-4/checkpoint-1000"
export DATASET_ID="downloaded_datatset/HumanEdit"
# export OUTPUT_MODEL="finetuned_models/ip2p_nollm_res256_lr5e-5_pretrained_unet_1000steps_13laststeps"
export OUTPUT_MODEL="finetuned_models/ip2p_llm_start0.9_des0.5_den0.5_res256_lr5e-4_pretrained_unet_1000steps_4nextepochs"
export TARGET_PROMPT="OUTPUT_CAPTION_BY_LLAMA"

CUDA_VISIBLE_DEVICES="1,4" accelerate launch --mixed_precision="fp16" finetuning.py \
 --pretrained_model_name_or_path=$MODEL_NAME \
 --pretrained_unet_name_or_path=$UNET_MODEL_PATH \
 --dataset_name=$DATASET_ID \
 --use_ema \
 --use_LLM_feedback=True --LLM_start_ratio=0 \
 --enable_xformers_memory_efficient_attention \
 --resolution=256 --random_flip \
 --train_batch_size=64 --gradient_accumulation_steps=4 --gradient_checkpointing \
 --target_prompt_column=$TARGET_PROMPT \
 --num_train_epochs=4 \
 --checkpointing_steps=10 --checkpoints_total_limit=10 \
 --learning_rate=5e-05 --lr_warmup_steps=0 \
 --conditioning_dropout_prob=0.05 \
 --mixed_precision=fp16 \
 --seed=42 \
 --output_dir=$OUTPUT_MODEL