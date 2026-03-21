#!/usr/bin/env python3
"""
从训练日志中提取 Open World 评估指标并可视化。
扩展支持 Absolute OSE 和 Wilderness Impact (WI) at Recall=0.8。

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


def parse_absolute_ose(log_path):
    """
    返回 Absolute OSE (IoU=50) 值列表，按出现顺序。
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = r'Absolute OSE \(IoU=50\): (\d+\.?\d*)'
    matches = re.findall(pattern, content)
    return [float(m) for m in matches]


def parse_wi_tables(log_path):
    """
    返回 Wilderness Impact (WI) 表格列表，每个元素为 dict {recall: wi_value}。
    只提取 Recall 和 WI-IoU50 列。
    """
    tables = []
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 检测表头行：必须包含 "Recall" 和 "WI-IoU50"（可能带连字符或下划线）
        # 同时过滤掉可能的分隔线（如 "--------" 或 "---"）
        if ('Recall' in line and 'WI-IoU50' in line and not line.startswith('-')):
            # 跳过可能的分隔线（下一行可能是 "--------" 或 "---"）
            i += 1
            # 跳过连续的分隔线
            while i < len(lines) and lines[i].strip().startswith('-'):
                i += 1
            # 开始解析数据行
            table = {}
            while i < len(lines):
                data_line = lines[i].strip()
                # 遇到空行或下一张表头则结束当前表格
                if not data_line or ('Recall' in data_line and 'WI-IoU50' in data_line):
                    break
                # 尝试按空白分割（支持空格/制表符）
                parts = data_line.split()
                if len(parts) >= 3:
                    try:
                        recall = float(parts[0])
                        wi = float(parts[1])
                        table[recall] = wi
                    except ValueError:
                        pass
                i += 1
            if table:
                tables.append(table)
        else:
            i += 1
    return tables


def extract_metrics(group_tables, absolute_ose_list, wi_tables_list):
    """
    从解析出的表格和指标列表中提取需要的指标序列。
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
        'Absolute_OSE': [],
        'WI_Recall0.8': [],
    }

    n_groups = len(group_tables)
    n_ose = len(absolute_ose_list)
    n_wi = len(wi_tables_list)

    # 长度对齐：如果长度不一致，截断到最小长度并给出警告
    min_len = min(n_groups, n_ose, n_wi)
    if n_groups != n_ose or n_groups != n_wi:
        print(f"警告：Group表格数量({n_groups})、Absolute OSE数量({n_ose})、WI表格数量({n_wi})不一致。")
        print(f"将仅使用前 {min_len} 个评估步骤进行绘图。")

    for idx in range(min_len):
        tbl = group_tables[idx]

        # 提取 Recall
        for group in needed_groups_recall:
            if group in tbl:
                _, _, recall50 = tbl[group]
                metrics[f'Recall_{group}'].append(recall50)
            else:
                metrics[f'Recall_{group}'].append(np.nan)

        # 提取 AP50
        for group in needed_groups_ap50:
            if group in tbl:
                ap50, _, _ = tbl[group]
                metrics[f'AP50_{group}'].append(ap50)
            else:
                metrics[f'AP50_{group}'].append(np.nan)

        # 提取 Absolute OSE
        metrics['Absolute_OSE'].append(absolute_ose_list[idx])

        # 提取 WI at Recall=0.8
        wi_table = wi_tables_list[idx]
        target_recall = 0.8
        # 查找精确匹配，否则取最接近的 recall 值
        if target_recall in wi_table:
            metrics['WI_Recall0.8'].append(wi_table[target_recall])
        else:
            # 查找最接近的 recall 键
            recalls = np.array(list(wi_table.keys()))
            if len(recalls) > 0:
                closest_idx = np.argmin(np.abs(recalls - target_recall))
                closest_recall = recalls[closest_idx]
                metrics['WI_Recall0.8'].append(wi_table[closest_recall])
                if abs(closest_recall - target_recall) > 1e-4:
                    print(f"警告：未找到精确的 Recall={target_recall}，使用最接近的 {closest_recall}")
            else:
                metrics['WI_Recall0.8'].append(np.nan)

    return metrics

def plot_metrics(metrics, save_path):
    """
    绘制三条曲线（Recall、AP50、Absolute OSE + WI）并保存。
    第三子图使用双Y轴：左轴为Absolute OSE，右轴为WI@Recall0.8。
    """
    if not metrics or all(len(v) == 0 for v in metrics.values()):
        print("没有有效指标可绘制。")
        return

    epochs = range(1, len(next(iter(metrics.values()))) + 1)

    # 创建三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

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
    ax2.set_ylabel('AP50 (%)')
    ax2.set_title('AP50 per Group')
    ax2.legend()
    ax2.grid(True)

    # 子图3：Absolute OSE 和 WI (双Y轴)
    ax3.set_xlabel('Evaluation Step')
    ax3.set_title('Additional Metrics (Absolute OSE and WI)')
    ax3.grid(True)

    # 左轴：Absolute OSE
    color_left = 'tab:red'
    ax3.set_ylabel('Absolute OSE (IoU=50)', color=color_left)
    if 'Absolute_OSE' in metrics and metrics['Absolute_OSE']:
        ax3.plot(epochs, metrics['Absolute_OSE'], marker='^', color=color_left, label='Absolute OSE (IoU=50)')
    ax3.tick_params(axis='y', labelcolor=color_left)

    # 右轴：WI@Recall0.8
    ax3_right = ax3.twinx()
    color_right = 'tab:blue'
    ax3_right.set_ylabel('WI@Recall0.8', color=color_right)
    if 'WI_Recall0.8' in metrics and metrics['WI_Recall0.8']:
        ax3_right.plot(epochs, metrics['WI_Recall0.8'], marker='d', color=color_right, label='WI@Recall0.8')
    ax3_right.tick_params(axis='y', labelcolor=color_right)

    # 合并图例（避免重复）
    lines_left, labels_left = ax3.get_legend_handles_labels()
    lines_right, labels_right = ax3_right.get_legend_handles_labels()
    if lines_left or lines_right:
        ax3.legend(lines_left + lines_right, labels_left + labels_right, loc='upper left')

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

    # 解析 Group-wise metrics 表格
    group_tables = parse_log_file(args.log_file)
    if not group_tables:
        sys.exit(1)
    print(f"找到 {len(group_tables)} 个 Group-wise 评估表格")

    # 解析 Absolute OSE
    absolute_ose_list = parse_absolute_ose(args.log_file)
    print(f"找到 {len(absolute_ose_list)} 个 Absolute OSE 值")

    # 解析 WI 表格
    wi_tables_list = parse_wi_tables(args.log_file)
    print(f"找到 {len(wi_tables_list)} 个 WI 表格")

    # 提取所有指标
    metrics = extract_metrics(group_tables, absolute_ose_list, wi_tables_list)

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