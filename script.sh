#!/bin/bash

#SBATCH --job-name=create_description
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx-small
#SBATCH --output=train_outs/gen_data/out/%x.%j.out
#SBATCH --error=train_outs/gen_data/errors/%x.%j.err
#SBATCH --mail-type=ALL

CUDA_VISIBLE_DEVICES="3" python create_target_description.py

# Model and dataset paths
# export MODEL_NAME="pretrained_frameworks/pretrained_IEDMs/instruct-pix2pix"
# export UNET_MODEL_PATH="finetuned_models/ip2p_nollm_res256_lr5e-4/checkpoint-1000"
# export DATASET_ID="downloaded_datatset/HumanEdit"
# # export OUTPUT_MODEL="finetuned_models/ip2p_nollm_res256_lr5e-5_pretrained_unet_1000steps_13laststeps"
# export OUTPUT_MODEL="finetuned_models/ip2p_llm_start0.9_des0.5_den0.5_res256_lr5e-4_pretrained_unet_1000steps_4nextepochs"
# export TARGET_PROMPT="OUTPUT_CAPTION_BY_LLAMA"

# CUDA_VISIBLE_DEVICES="5" accelerate launch --mixed_precision="fp16" finetuning_copy.py \
#  --pretrained_model_name_or_path=$MODEL_NAME \
#  --pretrained_unet_name_or_path=$UNET_MODEL_PATH \
#  --dataset_name=$DATASET_ID \
#  --use_ema \
#  --use_LLM_feedback=False --LLM_start_ratio=1 \
#  --use_localize_loss=True --mask_threshold=0.7 --exclude_layers_index_list=0,8 --timestep_threshold=500\
#  --enable_xformers_memory_efficient_attention \
#  --resolution=256 --random_flip \
#  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing \
#  --target_prompt_column=$TARGET_PROMPT \
#  --num_train_epochs=4 \
#  --checkpointing_steps=1000 --checkpoints_total_limit=10 \
#  --learning_rate=5e-05 --lr_warmup_steps=0 \
#  --conditioning_dropout_prob=0.05 \
#  --mixed_precision=fp16 \
#  --seed=42 \
#  --output_dir=$OUTPUT_MODEL
