#!/bin/sh

MODEL_ID='claudiom4sir/StableVSR'
OUTPUT_DIR='experiments/test_run_experiment'

# modify the GPU indices we use
GPUS="7"

GPUS_STR=$(echo $GPUS | tr ' ' ',')

export CUDA_VISIBLE_DEVICES=$GPUS_STR

# Calculate the number of GPUs (i.e., the number of processes)
NUM_PROCESSES=$(echo $GPUS | wc -w)


# Should restore in the real training: validation_steps=1000, max_train_steps=20000
# only set resume_from_checkpoint when need to retrain from some checkpoint
accelerate launch --num_processes $NUM_PROCESSES --main_process_port 29501 train.py \
 --pretrained_model_name_or_path=$MODEL_ID \
 --pretrained_vae_model_name_or_path=$MODEL_ID \
 --output_dir=$OUTPUT_DIR \
 --dataset_config_path="/home/yuxin/StableVSR/dataset/config_reds.yaml" \
 --learning_rate=5e-5 \
 --validation_steps=1000 \
 --train_batch_size=8 \
 --dataloader_num_workers=8 \
 --max_train_steps=20000 \
 --enable_xformers_memory_efficient_attention \
 --validation_image "/home/yuxin/StableVSR/Datasets/REDS/val/val_blur/020/00000000.png;/home/yuxin/StableVSR/Datasets/REDS/val/val_blur/020/00000001.png;/home/yuxin/StableVSR/Datasets/REDS/val/val_blur/020/00000002.png;/home/yuxin/StableVSR/Datasets/REDS/val/val_blur/020/00000003.png;/home/yuxin/StableVSR/Datasets/REDS/val/val_blur/020/00000004.png;/home/yuxin/StableVSR/Datasets/REDS/val/val_blur/020/00000005.png;/home/yuxin/StableVSR/Datasets/REDS/val/val_blur/020/00000006.png;/home/yuxin/StableVSR/Datasets/REDS/val/val_blur/020/00000007.png;/home/yuxin/StableVSR/Datasets/REDS/val/val_blur/020/00000008.png;/home/yuxin/StableVSR/Datasets/REDS/val/val_blur/020/00000009.png"  \
 --validation_prompt "" \
 --resume_from_checkpoint="/home/yuxin/StableVSR/experiments/test_run_experiment/checkpoint-7000"
