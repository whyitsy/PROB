#!/usr/bin/env bash

echo "Starting Ablation Study on M-OWODB Dataset (Task 1 Only)"

set -x
set -e

BASE_EXP_DIR=/mnt/data/kky/output/PROB/exps/MOWODB/ABLATION_STUDY
PY_ARGS=${@:1}

# 定义四个阶段的消融配置
# 格式: "实验名称|额外的参数控制"
CONFIGS=(
    "Exp4_Innov1_FullVSAD|--model_type innov_1 --enable_unk_label_obj --obj_loss_coef 1000 --etop --tdqi --use_valid_mask --use_feature_align --align_loss_coef 2.0 --use_vlm_distill"
    
    "Exp2_Innov1_HardLabel|--model_type innov_1 --enable_unk_label_obj --obj_loss_coef 1000 --etop --tdqi --use_valid_mask"
    
    "Exp3_Innov1_DummyLabel|--model_type innov_1 --enable_unk_label_obj --obj_loss_coef 1000"
    
    "Exp1_Baseline|--model_type prob"
)

# 循环执行消融实验
for CONFIG in "${CONFIGS[@]}"; do
    # 解析名称和参数
    EXP_NAME="${CONFIG%%|*}" # %% 是 Bash 的“从尾部开始最大匹配删除”操作符。|* 表示模式：一个竖线后跟任意字符。
    EXTRA_ARGS="${CONFIG#*|}" # # 是 Bash 的“从开头开始最小匹配删除”操作符。*| 表示模式：任意字符后跟一个竖线。
    
    CURRENT_OUT_DIR="${BASE_EXP_DIR}/${EXP_NAME}/t1"
    WANDB_NAME="Ablation_${EXP_NAME}"
    
    echo "=========================================================="
    echo "Running Ablation: ${EXP_NAME}"
    echo "Arguments: ${EXTRA_ARGS}"
    echo "=========================================================="
    
    export WANDB_MODE=offline
    export WANDB_DIR=${CURRENT_OUT_DIR}
    
    python -u main_open_world.py \
        --output_dir "${CURRENT_OUT_DIR}" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
        --train_set 'owod_t1_train' --test_set 'owod_all_task_test' --epochs 41 --obj_temp 1.3 \
        --wandb_name "${WANDB_NAME}_t1" --exemplar_replay_selection --exemplar_replay_max_length 850 \
        --exemplar_replay_dir ${WANDB_NAME} --exemplar_replay_cur_file "learned_owod_t1_ft.txt" \
        --lr 1e-4 \
        ${EXTRA_ARGS} \
        ${PY_ARGS}
        
    echo "Finished ${EXP_NAME}."
    echo "----------------------------------------------------------"
done

echo "All ablation studies for Task 1 are completed!"