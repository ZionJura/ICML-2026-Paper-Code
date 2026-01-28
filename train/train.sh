#!/bin/bash
set -e
trap 'echo "Ctrl+C detected. Exiting..."; exit 1' SIGINT

# ============================================================
# 0. 环境初始化（只做一次）
# ============================================================
# source /mnt/home/venv/bin/activate

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
ulimit -u 100000
ulimit -n 100000

# ============================================================
# 1. 运行模式控制
#   train_and_eval | eval_only
# ============================================================
MODE="train_and_eval"
# MODE="eval_only"

# ============================================================
# 2. 实验列表（每行一个实验）
#   OUTPUT_DIR##MODEL_NAME_OR_PATH##APPEND_FILES_DIR
# ============================================================
exp_list=(
    
    "../models/sft_checkpoint/timesense_stage2##../models/sft_checkpoint/timesense_stage0##../scripts/TimeSense/append_2stage"
)

# ============================================================
# 3. 固定训练参数（全局共用，不再重复）
# ============================================================
TRAIN_WORKDIR="../ChatTS-Training"
EVAL_WORKDIR="..//evaluation"
DS_CONFIG="./ds_config_2_bf16_offload.json"

DATASET="generated_timeseries_bench_plus_polished,sft,ift"
INTERLEAVE_PROBS="0.7,0.2,0.1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_WORKDIR="$(realpath "$SCRIPT_DIR/${TRAIN_WORKDIR}")"
EVAL_WORKDIR="$(realpath "$SCRIPT_DIR/${EVAL_WORKDIR}")"
DS_CONFIG="$(realpath "$SCRIPT_DIR/${DS_CONFIG}")"


# ============================================================
# 4. 函数定义
# ============================================================

copy_append_files() {
    local append_dir=$1
    local model_dir=$2
    echo "[INFO] Copy append files: $append_dir -> $model_dir"
    for file in "$append_dir"/*; do
        cp -f "$file" "$model_dir/"
    done
}

save_script_snapshot() {
    local output_dir=$1
    local script_path="${BASH_SOURCE[0]}"
    mkdir -p "$output_dir"
    cp "$script_path" "$output_dir/$(basename "$script_path")"
}

train() {
    local OUTPUT_DIR=$1
    local MODEL_NAME_OR_PATH=$2

    echo "[TRAIN] Start training: $OUTPUT_DIR"

    cd "$TRAIN_WORKDIR"

    DEEPSPEED_TIMEOUT=120 deepspeed --num_gpus 8 --master_port=19901 src/train.py \
        --deepspeed "$DS_CONFIG" \
        --stage sft \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --dataset "$DATASET" \
        --interleave_probs "$INTERLEAVE_PROBS" \
        --do_train \
        --mix_strategy "interleave_over" \
        --template "chatts" \
        --finetuning_type full \
        --output_dir "$OUTPUT_DIR" \
        --overwrite_output_dir \
        --trust_remote_code True \
        --report_to "none" \
        --full_determinism True \
        --seed 66 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 32 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --save_steps 600 \
        --learning_rate 1e-5 \
        --warmup_ratio 0.02 \
        --num_train_epochs 0 \
        --max_steps 1200 \
        --bf16 \
        --plot_loss \
        --save_only_model \
        --save_safetensors False \
        --preprocessing_num_workers 96 \
        --cutoff_len 12000
}

evaluate() {
    local OUTPUT_DIR=$1
    local APPEND_FILES_DIR=$2

    echo "[EVAL] Start evaluation: $OUTPUT_DIR"

    cd "$EVAL_WORKDIR"

    bash generate.sh \
        "$OUTPUT_DIR" \
        "$(basename "$OUTPUT_DIR")" \
        "$APPEND_FILES_DIR"

    sleep 120
}

run_one_experiment() {
    local exp_str=$1

    OUTPUT_DIR=$(echo "$exp_str" | awk -F '##' '{print $1}')
    MODEL_NAME_OR_PATH=$(echo "$exp_str" | awk -F '##' '{print $2}')
    APPEND_FILES_DIR=$(echo "$exp_str" | awk -F '##' '{print $3}')


    if [[ -z "$OUTPUT_DIR" || -z "$MODEL_NAME_OR_PATH" || -z "$APPEND_FILES_DIR" ]]; then
        echo "[ERROR] Invalid experiment string:"
        echo "  $exp_str"
        echo "Parsed as:"
        echo "  OUTPUT_DIR=$OUTPUT_DIR"
        echo "  MODEL_NAME_OR_PATH=$MODEL_NAME_OR_PATH"
        echo "  APPEND_FILES_DIR=$APPEND_FILES_DIR"
        exit 1
    fi


    copy_append_files "$APPEND_FILES_DIR" "$MODEL_NAME_OR_PATH"
    save_script_snapshot "$OUTPUT_DIR"

    if [[ "$MODE" == "train_and_eval" ]]; then
        train "$OUTPUT_DIR" "$MODEL_NAME_OR_PATH"
        evaluate "$OUTPUT_DIR" "$APPEND_FILES_DIR"
    elif [[ "$MODE" == "eval_only" ]]; then
        evaluate "$OUTPUT_DIR" "$APPEND_FILES_DIR"
    else
        echo "[ERROR] Unknown MODE=$MODE"
        exit 1
    fi
}

for exp in "${exp_list[@]}"; do
    rm -rf ~/.cache/huggingface
    run_one_experiment "$exp"
    sleep 120
done

echo "[DONE] All experiments finished."
sleep 120
python /zzr/matric_calc2.py