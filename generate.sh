
NUM_GPUS=8  # 总GPU数量
MIN_MEMORY=10000  # 所需最小显存(MB)，可根据实际需求调整

LOG_DIR="./results"

declare -a GPU_USED
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_USED[$i]=0
done

declare -a TASK_QUEUE
declare -a RUNNING_TASKS 
TOTAL_TASKS=0
COMPLETED_TASKS=0
SCHEDULER_PID=$$ 
cleanup() {
    echo -e "\n收到中断信号(ctrl+c)，正在终止所有任务..."
    
    if [ ${#RUNNING_TASKS[@]} -gt 0 ]; then
        for task in "${RUNNING_TASKS[@]}"; do
            pid=$(echo "$task" | cut -d: -f3 2>/dev/null)
            lab_name=$(echo "$task" | cut -d: -f2 2>/dev/null)
            
            if [[ "$pid" =~ ^[0-9]+$ ]]; then
                echo "终止任务 $lab_name (PID: $pid)..."
                kill $pid 2>/dev/null
                wait $pid 2>/dev/null &
                sleep 10
                if ps -p $pid > /dev/null; then
                    kill -9 $pid 2>/dev/null
                    echo "强制终止任务 $lab_name (PID: $pid)"
                fi
            else
                echo "无效的PID: $pid，跳过终止"
            fi
        done
    fi
    
    echo "所有任务已终止，正在退出..."
    exit 0
}

trap cleanup SIGINT SIGTERM

show_progress() {
    clear
    echo "===== 任务调度状态 ====="
    echo "总任务数: $TOTAL_TASKS"
    echo "已完成: $COMPLETED_TASKS"
    echo "运行中: $((${#RUNNING_TASKS[@]}))"
    echo "等待中: $((${#TASK_QUEUE[@]}))"
    echo "------------------------"
    echo "按 Ctrl+C 终止所有任务并退出"
    echo "------------------------"
    
    echo "运行中任务:"
    if [ ${#RUNNING_TASKS[@]} -eq 0 ]; then
        echo "  无"
    else
        for task in "${RUNNING_TASKS[@]}"; do
            gpu_id=$(echo "$task" | cut -d: -f1 2>/dev/null)
            lab_name=$(echo "$task" | cut -d: -f2 2>/dev/null)
            pid=$(echo "$task" | cut -d: -f3 2>/dev/null)
            gen_log=$(echo "$task" | cut -d: -f4 2>/dev/null)
            eval_log=$(echo "$task" | cut -d: -f5 2>/dev/null)
            
            if [[ -n "$gpu_id" && -n "$lab_name" && -n "$pid" ]]; then
                echo "GPU $gpu_id: $lab_name (PID: $pid)"
                echo "  生成日志: $gen_log"
                echo "  评估日志: $eval_log"
            else
                echo "  无效任务记录: $task"
            fi
            echo "  ------------------------"
        done
    fi
    
    echo "GPU状态:"
    for ((i=0; i<NUM_GPUS; i++)); do
        if [ ${GPU_USED[$i]} -eq 0 ]; then
            echo -n "GPU $i: 空闲  "
        else
            echo -n "GPU $i: 占用  "
        fi
        if (( (i+1) % 4 == 0 )); then
            echo
        fi
    done
    echo -e "\n========================\n"
}

get_available_gpu() {
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        if [ ${GPU_USED[$gpu]} -eq 0 ]; then
            mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu 2>/dev/null)
            if [ $? -eq 0 ] && [[ "$mem" =~ ^[0-9]+$ ]] && [ $mem -ge $MIN_MEMORY ]; then
                echo $gpu
                return 0
            fi
        fi
    done
    echo "-1"  # 无可用GPU
    return 1
}

release_gpu() {
    local gpu_id=$1
    if [[ "$gpu_id" =~ ^[0-9]+$ ]] && [ $gpu_id -ge 0 ] && [ $gpu_id -lt $NUM_GPUS ]; then
        GPU_USED[$gpu_id]=0
        
        if [ ${#RUNNING_TASKS[@]} -gt 0 ]; then
            for i in "${!RUNNING_TASKS[@]}"; do
                if [[ "${RUNNING_TASKS[$i]}" == "${gpu_id}:"* ]]; then
                    unset "RUNNING_TASKS[$i]"
                    RUNNING_TASKS=("${RUNNING_TASKS[@]}")
                    break
                fi
            done
        fi
        
        COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
        show_progress
    fi
}

start_task() {
    local lab_name=$1
    local gpu_id=$2
    
    if [[ -z "$lab_name" || ! "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "无效的任务参数: lab_name=$lab_name, gpu_id=$gpu_id"
        return 1
    fi
    
    local save_folder="${LOG_DIR}/${folder_name}"
    local gen_log="${save_folder}/generating_${lab_name}.log"
    local eval_log="${save_folder}/${lab_name}.log"
    
    GPU_USED[$gpu_id]=1
    
    echo "Starting task: GPU=${gpu_id}, lab_name=${lab_name}, model=${model_name}"
    
    if [ ! -d "$save_folder" ]; then
        echo "Creating log directory: $save_folder ..."
        mkdir -p "$save_folder"
        if [ $? -ne 0 ]; then
            echo "Failed to create log directory. Exiting..."
            exit 1
        fi
    fi

    (
        trap 'exit 0' SIGINT SIGTERM
        
        model_base=$(basename "$model_name")

        CUDA_VISIBLE_DEVICES="${gpu_id}" python evaluate.py \
            --lab_name "${lab_name}" \
            --model_name "${model_name}" \
            --save_folder "${save_folder}" > "$gen_log" 2>&1
        
        if [ $? -ne 0 ]; then
            echo "Error: compare_gty.py failed for $lab_name on GPU $gpu_id" >> "${save_folder}/errors.log"
        else
            CUDA_VISIBLE_DEVICES="${gpu_id}" python calculate_metric.py \
                --model_qa_name "${model_name}" \
                --model_name "${folder_name}" \
                --evaluation_datasets "${lab_name}" \
                --save_folder "${save_folder}" > "${eval_log}" 2>&1
            
            if [ $? -ne 0 ]; then
                echo "Error: calculate_metrix.py failed for $lab_name on GPU $gpu_id" >> "${save_folder}/errors.log"
            fi
        fi
        
        release_gpu $gpu_id
    ) &
    
    local child_pid=$!
    if [[ "$child_pid" =~ ^[0-9]+$ ]]; then
        RUNNING_TASKS+=("${gpu_id}:${lab_name}:${child_pid}:${gen_log}:${eval_log}")
        show_progress
    else
        echo "创建任务失败: 无效的子进程PID"
        release_gpu $gpu_id
    fi
}

schedule_tasks() {
    while true; do
        if [ ${#RUNNING_TASKS[@]} -gt 0 ]; then
            local tasks_copy=("${RUNNING_TASKS[@]}")
            for i in "${!tasks_copy[@]}"; do
                local task="${tasks_copy[$i]}"
                local pid=$(echo "$task" | cut -d: -f3 2>/dev/null)
                local gpu_id=$(echo "$task" | cut -d: -f1 2>/dev/null)
                
                if [[ "$pid" =~ ^[0-9]+$ ]]; then
                    if ! ps -p "$pid" > /dev/null 2>&1; then
                        release_gpu "$gpu_id"
                    fi
                else
                    echo "清理无效任务记录（PID无效）: $task"
                    release_gpu "$gpu_id"
                fi
            done
        fi
        
        if [ ${#TASK_QUEUE[@]} -gt 0 ]; then
            gpu_id=$(get_available_gpu)
            if [ "$gpu_id" != "-1" ] && [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
                lab_name="${TASK_QUEUE[0]}"
                TASK_QUEUE=("${TASK_QUEUE[@]:1}")
                start_task "$lab_name" "$gpu_id"
            fi
        else
            if [ ${#RUNNING_TASKS[@]} -eq 0 ]; then
                echo "所有任务已完成！"
                break
            fi
        fi
        
        sleep 2  # 短暂等待后再次检查
    done
}

main() {
    model_name="$1"
    folder_name="$2"
    
    echo "Starting compare_gty.py tasks $model_name at $folder_name"
    
    if [ $# -ge 3 ]; then
        APPEND_FILES_DIR="$3"
        echo "Copying files from $APPEND_FILES_DIR to $model_name"
        if [ -d "$APPEND_FILES_DIR" ]; then
            for file in "$APPEND_FILES_DIR"/*; do
                if [ -f "$file" ]; then  # 只复制文件
                    cp -f "$file" "$model_name/"
                fi
            done
            echo "Files copied successfully."
        else
            echo "警告: 源目录 $APPEND_FILES_DIR 不存在，跳过文件复制"
        fi
    fi
    
    if [ -z "$model_name" ] || [ -z "$folder_name" ]; then
        echo "Usage: $0 <model_name> <folder_name> [append_files_dir]"
        exit 1
    fi
    
    if [ ! -d "$LOG_DIR" ]; then
        echo "Log directory does not exist. Creating $LOG_DIR ..."
        mkdir -p "$LOG_DIR"
        if [ $? -ne 0 ]; then
            echo "Failed to create log directory. Exiting..."
            exit 1
        fi
    fi
    
    TASK_QUEUE=(
        "tms_ift_task"
        "tms_uni_extreme_task"
        "tms_multi_extreme_task"
        "tms_uni_change_point_task"
        "tms_multi_change_point_task"
        "tms_uni_trend_task"
        "tms_multi_trend_task"
        "tms_uni_spike_task"
        "tms_multi_spike_task"
        "tms_uni_period_task"
        "tms_multi_period_task"
        "tms_comparison_task"
        "tms_segment_task"
        "tms_relative_task"
        "tms_anomaly_detection_task"
        "tms_root_cause_analysis_task"
        "tms_describe_task"
    )
    
    TOTAL_TASKS=${#TASK_QUEUE[@]}
    COMPLETED_TASKS=0
    
    echo "总任务数: $TOTAL_TASKS"
    echo "开始任务调度..."
    show_progress
    schedule_tasks


}

main "$@"
