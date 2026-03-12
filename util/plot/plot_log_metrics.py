#!/usr/bin/env python3
"""
从训练日志中提取 Open World 评估指标并可视化。

使用方法：
    python parse_log.py /path/to/logfile

要求：
    Python 3.6+，matplotlib 和 numpy 会自动安装（如缺失）。
"""

import os
import re
import sys
import argparse
import subprocess
from collections import defaultdict

# 尝试导入绘图库，若缺失则自动安装
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("缺少 matplotlib 或 numpy，正在尝试安装...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "numpy"])
    import matplotlib.pyplot as plt
    import numpy as np


def parse_log_file(log_path):
    """
    读取日志文件，提取所有 Group-wise metrics 表格。
    返回列表，每个元素为 dict {group: (ap50, prec50, recall50)}
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正则匹配整个表格块：以 "| Group" 开头，后面跟着若干以 "|" 开头的行
    pattern = r'^(\| Group.*\|.*\n(?:\|.*\|.*\n)*)'
    matches = re.findall(pattern, content, re.MULTILINE)

    if not matches:
        print(f"错误：在日志文件中未找到任何 Group-wise metrics 表格。")
        return []

    tables = []
    for block in matches:
        lines = block.strip().split('\n')
        # 跳过空行
        lines = [ln for ln in lines if ln.strip()]
        if len(lines) < 2:  # 至少表头 + 一行数据
            continue

        # 解析数据行
        data = {}
        for line in lines[1:]:  # 跳过表头
            if not line.startswith('|'):
                continue
            # 按 '|' 分割，去除首尾空单元格
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 4:
                group = parts[0]
                try:
                    ap50 = float(parts[1])
                    prec50 = float(parts[2])
                    recall50 = float(parts[3])
                    data[group] = (ap50, prec50, recall50)
                except ValueError:
                    # 若转换失败则跳过该行
                    continue
        if data:
            tables.append(data)

    return tables


def extract_metrics(tables):
    """
    从解析出的表格列表中提取需要的指标序列。
    返回字典，键为指标名，值为列表。
    """
    # 所需指标
    needed_groups_recall = {'Current', 'Known', 'Unknown'}
    needed_groups_ap50 = {'Current', 'Known'}

    # 初始化存储
    metrics = {
        'Recall_Current': [],
        'Recall_Known': [],
        'Recall_Unknown': [],
        'AP50_Current': [],
        'AP50_Known': [],
    }

    for tbl in tables:
        # 提取 Recall
        for group in needed_groups_recall:
            if group in tbl:
                ap50, prec50, recall50 = tbl[group]
                metrics[f'Recall_{group}'].append(recall50)
            else:
                # 若某次评估缺少该组，用 NaN 填充
                metrics[f'Recall_{group}'].append(np.nan)

        # 提取 AP50
        for group in needed_groups_ap50:
            if group in tbl:
                ap50, prec50, recall50 = tbl[group]
                metrics[f'AP50_{group}'].append(ap50)
            else:
                metrics[f'AP50_{group}'].append(np.nan)

    return metrics


def plot_metrics(metrics, save_path):
    """
    绘制两条曲线（Recall 和 AP50）并保存。
    """
    if not metrics or all(len(v) == 0 for v in metrics.values()):
        print("没有有效指标可绘制。")
        return

    epochs = range(1, len(next(iter(metrics.values()))) + 1)

    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 子图1：Recall
    for group in ['Current', 'Known', 'Unknown']:
        key = f'Recall_{group}'
        if key in metrics and metrics[key]:
            ax1.plot(epochs, metrics[key], marker='o', label=f'{group} Recall')
    ax1.set_ylabel('Recall (%)')
    ax1.set_title('Recall per Group')
    ax1.legend()
    ax1.grid(True)

    # 子图2：AP50
    for group in ['Current', 'Known']:
        key = f'AP50_{group}'
        if key in metrics and metrics[key]:
            ax2.plot(epochs, metrics[key], marker='s', label=f'{group} AP50')
    ax2.set_xlabel('Evaluation Step')
    ax2.set_ylabel('AP50 (%)')
    ax2.set_title('AP50 per Group')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"图片已保存至：{save_path}")
    plt.close()


def get_base_name_from_log(log_path):
    """
    从日志文件路径中提取两级父目录名拼接作为基础名。
    例如：/path/to/exp1/logs/train.log -> exp1_logs
    """
    log_path = os.path.normpath(log_path)
    parts = []
    # 第一级父目录
    parent1 = os.path.basename(os.path.dirname(log_path))
    if parent1:
        parts.append(parent1)
    # 第二级父目录
    parent2 = os.path.basename(os.path.dirname(os.path.dirname(log_path)))
    if parent2 and parent2 != parent1:  # 防止重复
        parts.insert(0, parent2)  # 保证顺序：父级在前
    if not parts:
        # 若没有父目录，使用日志文件名（不含扩展名）
        base = os.path.splitext(os.path.basename(log_path))[0]
    else:
        base = '_'.join(parts)
    return base


def main():
    parser = argparse.ArgumentParser(description='解析日志并绘制 Open World 评估指标')
    parser.add_argument('log_file', help='日志文件路径')
    args = parser.parse_args()

    if not os.path.isfile(args.log_file):
        print(f"错误：文件不存在 {args.log_file}")
        sys.exit(1)

    # 解析日志
    tables = parse_log_file(args.log_file)
    if not tables:
        sys.exit(1)

    print(f"找到 {len(tables)} 个评估表格")

    # 提取指标
    metrics = extract_metrics(tables)

    # 准备保存路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, os.path.splitext(os.path.basename(__file__))[0])
    os.makedirs(save_dir, exist_ok=True)

    base_name = get_base_name_from_log(args.log_file)
    save_path = os.path.join(save_dir, f"{base_name}_metrics.png")

    # 绘图
    plot_metrics(metrics, save_path)


if __name__ == '__main__':
    main()