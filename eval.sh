#!/bin/bash
#SBATCH -o /aifs4su/rubickjiang/logs/job.%j.out.log
#SBATCH --error /aifs4su/rubickjiang/logs/job.%j.err.log
#SBATCH -p batch
#SBATCH -J test_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:8
#SBATCH -c 32
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MACLAB_NAS_NAME="maclabcv2"

MODEL_PATH="/mnt/maclabcv2/rubickjiang/proj_storage/unlearning/MT_models/wmt_zh/checkpoint-313"
DATA_PATH=""
DATASET_NAME="qasc"
SPLIT=0
VALID_DATA_PATH=""
OUTPUT_DIR=""
TEMP_PATH=""
PEFT_MODEL=""
export TORCH_USE_CUDA_DSA=1
# export CUDA_VISIBLE_DEVICES=0
# what matters: model_name_or_path, peft_model_path, eval_data_path, per_device_eval_batch_size(fixed)
export SEED=114514
TEST_DATA_NAMES=("wmt_zh" "qasc" "sst5" "medmcqa_dental" "mmlu_psychology")
for i in 0
do
accelerate launch --config_file "/mnt/${MACLAB_NAS_NAME}/rubickjiang/codes/accelerate_config/config_acc.yaml" evaluation.py \
    --model_name_or_path "$MODEL_PATH" \
    --mode "base" \
    --peft_model_path "" \
    --dataset_name "${TEST_DATA_NAMES[i]}" \
    --data_path "" \
    --valid_data_path "" \
    --eval_data_path "${TEST_DATA_NAMES[i]}:test" \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --lazy_preprocess False \
    --use_lora True \
    --gradient_checkpointing True
done

exit 0
# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json