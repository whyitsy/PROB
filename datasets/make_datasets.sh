#!/bin/bash

# PROB/data/OWOD目录
owod_dir="/gemini/code/PROB/data/OWOD/"

# coco数据集的目录
coco_train_dir="/gemini/data-1/train2017/"
coco_val_dir="/gemini/data-1/val2017/"
coco_annotation_dir="/gemini/data-1/annotations/"

# VOC数据集的目录
voc_2007_dir="/gemini/data-2/VOCdevkit/VOC2007/"
voc_2012_dir="/gemini/data-2/VOCdevkit/VOC2012/"

# 链接图片 ln -f 覆盖已存在, 幂等性
find "$coco_train_dir" -type f -exec ln -sf {} "${owod_dir}JPEGImages" \+
find "$coco_val_dir" -type f -exec ln -sf {} "${owod_dir}JPEGImages" \+
find "${voc_2007_dir}JPEGImages" -type f -exec ln -sf {} "${owod_dir}JPEGImages" \+
find "${voc_2012_dir}JPEGImages" -type f -exec ln -sf {} "${owod_dir}JPEGImages" \+

# 标注文件
find "${voc_2007_dir}Annotations" -type f -exec ln -sf {} "${owod_dir}Annotations" \+
find "${voc_2012_dir}Annotations" -type f -exec ln -sf {} "${owod_dir}Annotations" \+

# coco数据集的标注文件处理
python datasets/coco2voc_optimized.py \
    "${coco_annotation_dir}instances_train2017.json" \
    "${coco_annotation_dir}instances_val2017.json" \
    "$owod_dir"