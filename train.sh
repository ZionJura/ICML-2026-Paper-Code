#!/bin/bash
# source /mnt/home/venv/bin/activate
source /mnt/home/zr/zzr_workplace/venv/local_chatts/bin/activate
trap 'echo "Ctrl+C detected. Exiting..."; exit 1' SIGINT
set -e
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# 临时增加最大进程数
ulimit -u 100000
ulimit -n 100000
countdown() {
  local SECONDS_LEFT=$((0 * 30 * 60))  
  while [ $SECONDS_LEFT -gt 0 ]; do
    HOURS=$(($SECONDS_LEFT / 3600))
    MINUTES=$((($SECONDS_LEFT % 3600) / 60))
    SECONDS=$(($SECONDS_LEFT % 60))
    printf "\r%02d:%02d:%02d" $HOURS $MINUTES $SECONDS
    sleep 1
    SECONDS_LEFT=$(($SECONDS_LEFT - 1))
  done
  echo ""
}
countdown
# 定义参数
export https_proxy=http://100.68.175.95:3128
export http_proxy=http://100.68.175.95:3128
OUTPUT_DIR="/mnt/home/zr/models/chatts_1204_ext_vc_cold_start"
MODEL_NAME_OR_PATH="/mnt/home/zr/models/chatts_1204_ext_vc"
# MODEL_NAME_OR_PATH="/mnt/home/xz/zr/LLaMA-Factory-zzr/script/ChatTS/sft_checkpoint/Qwen2.5-14B-Instruct"
# /mnt/home/zr/models/ChatTS-14B
APPEND_FILES_DIR="/zzr/scripts/TimeSense/append_cold_start_special_ex" 
echo "Copying files from $APPEND_FILES_DIR to $MODEL_NAME_OR_PATH"
for file in "$APPEND_FILES_DIR"/*; do
    cp -f "$file" "$MODEL_NAME_OR_PATH/"
done

CURRENT_SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_FILENAME=$(basename "$CURRENT_SCRIPT_PATH")
mkdir -p "$OUTPUT_DIR"
cp "$CURRENT_SCRIPT_PATH" "$OUTPUT_DIR/$SCRIPT_FILENAME"

echo "脚本 '$SCRIPT_FILENAME' 已成功复制到 '$OUTPUT_DIR'"
    # --dataset "chatts_stage1_algin_256,chatts_stage_ift,14B-0612" \
    # --interleave_probs "0.8,0.1,0.1" \
echo "Files copied successfully."
#cd /mnt/home/zr/xz/opsfm-xz
cd /zzr/scripts/ChatTS-Training

DEEPSPEED_TIMEOUT=120 deepspeed --num_gpus 8 --master_port=19901 src/train.py \
    --deepspeed /zzr/scripts/ChatTS-Training/ds_config/ds_config_2_offload_bf16.json \
    --stage sft \
    --model_name_or_path "$MODEL_NAME_OR_PATH"\
    --dataset "sft,ift,generated_timeseries_bench_qa_new_special_token,describe_cold_start" \
    --interleave_probs "0.1,0.1,0.2,0.6" \
    --do_train \
    --mix_strategy "interleave_over" \
    --template "chatts"  \
    --finetuning_type full \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --trust_remote_code True \
    --report_to "wandb" \
    --full_determinism True \
    --seed 66 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 600 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.02 \
    --num_train_epochs 0 \
    --max_steps 200 \
    --bf16 \
    --plot_loss \
    --save_only_model \
    --save_safetensors False \
    --preprocessing_num_workers 32 \
    --cutoff_len 10000
    
CURRENT_SCRIPT_PATH="${BASH_SOURCE[0]}"
# 提取脚本文件名
SCRIPT_FILENAME=$(basename "$CURRENT_SCRIPT_PATH")
cp "$CURRENT_SCRIPT_PATH" "$OUTPUT_DIR/$SCRIPT_FILENAME"
echo "Files copied successfully."
sleep 120
bash /zzr/scripts/train_scripts/gpro/train.sh /zzr/scripts/train_scripts/gpro/config_describe.yaml
sleep 120
python /zzr/matric_calc2.py
# cd /mnt/home/zr/LLaMA-Factory-zzr/script/ChatTS/evaluate
#bash generate.sh "${OUTPUT_DIR}" chatts2.5-base_ts_extend-0612-ck00 "${APPEND_FILES_DIR}"
# bash generate.sh "${OUTPUT_DIR}" "$(basename "${OUTPUT_DIR}")" "${APPEND_FILES_DIR}"
# sleep 30
# bash generate "${OUTPUT_DIR}/checkpoint-600" "$(basename "${OUTPUT_DIR}")-600steps" "${APPEND_FILES_DIR}"
# python /mnt/home/matric_calc.py