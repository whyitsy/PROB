#!/bin/bash
set -e  # 遇到错误立即退出

# 定义数据目录
owod_dir="/gemini/code/PROB/data/OWOD/"
coco_train_dir="/gemini/data-1/train2017/"
coco_val_dir="/gemini/data-1/val2017/"
coco_annotation_dir="/gemini/data-1/annotations/"
voc_2007_dir="/gemini/data-2/VOCdevkit/VOC2007/"
voc_2012_dir="/gemini/data-2/VOCdevkit/VOC2012/"

# 创建目标目录
mkdir -p "${owod_dir}JPEGImages" "${owod_dir}Annotations"

# 定义一个函数来执行 find 并报告错误（用于后台子shell）
run_find() {
    if ! find "$@" -type f -exec ln -sf {} "${owod_dir}JPEGImages" \;; then
        echo "Error in find for: $*" >&2
        exit 1
    fi
}

# ---------- 并发执行图片链接 ----------
{
    run_find "$coco_train_dir"
} &
pid1=$!

{
    run_find "$coco_val_dir"
} &
pid2=$!

{
    run_find "${voc_2007_dir}JPEGImages"
} &
pid3=$!

{
    run_find "${voc_2012_dir}JPEGImages"
} &
pid4=$!

# ---------- 并发执行标注链接 ----------
{
    if ! find "${voc_2007_dir}Annotations" -type f -exec ln -sf {} "${owod_dir}Annotations" \;; then
        echo "Error in find for VOC2007 Annotations" >&2
        exit 1
    fi
} &
pid5=$!

{
    if ! find "${voc_2012_dir}Annotations" -type f -exec ln -sf {} "${owod_dir}Annotations" \;; then
        echo "Error in find for VOC2012 Annotations" >&2
        exit 1
    fi
} &
pid6=$!

# ---------- 并发执行 COCO 标注转换 ----------
{
    python datasets/coco2voc_optimized.py \
        "${coco_annotation_dir}instances_train2017.json" \
        "${coco_annotation_dir}instances_val2017.json" \
        "$owod_dir"
} &
pid7=$!

# ---------- 等待所有后台任务完成并检查返回值 ----------
fail=0
for pid in $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7; do
    wait $pid || { echo "Job $pid failed"; fail=1; }
done

if [ $fail -eq 1 ]; then
    echo "Some tasks failed." >&2
    exit 1
else
    echo "All tasks completed successfully."
fi