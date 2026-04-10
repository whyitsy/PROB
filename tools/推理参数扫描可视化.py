import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# ================= 配置区域 =================
ROOT_DIR = "/mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL"          # 根目录路径
OUTPUT_DIR = "/mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL/plots"  # 图像输出目录
FOLDER_LIST = ["t1", "t2_ft", "t3_ft", "t4_ft"]    # 第一层文件夹名称列表
FINAL_STAGE_ONLY_MAP = "t4_ft"  # 最后阶段仅保留 mAP
# ===========================================

def parse_log(file_path):
    """解析 log.txt，返回四个指标的字典"""
    with open(file_path, 'r') as f:
        content = f.read()
    lines = content.splitlines()
    
    # 1. A-OSE
    aose_match = re.search(r'Absolute OSE \(IoU=50\):\s*([\d.]+)', content)
    aose = float(aose_match.group(1)) if aose_match else None
    
    # 2. WI (Recall=0.8 对应的 WI-IoU50)
    wi = None
    # 逐行解析 Recall/WI 表，兼容空格与分隔线格式差异
    in_recall_table = False
    for line in lines:
        stripped = line.strip()
        if 'Recall' in stripped and 'WI-IoU50' in stripped and 'Unk-IoU50' in stripped:
            in_recall_table = True
            continue
        if not in_recall_table:
            continue
        if not stripped:
            # 表格结束
            break
        if set(stripped) <= set('- '):
            # 跳过分隔线
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            recall = float(parts[0])
            wi_val = float(parts[1])
        except ValueError:
            continue
        if abs(recall - 0.8) < 1e-6:
            wi = wi_val
            break
    
    # 3. U-Recall (Group=Unknown 的 Recall50)
    urecall = None
    urecall_pattern = r'\|\s*Unknown\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
    urecall_match = re.search(urecall_pattern, content)
    if urecall_match:
        urecall = float(urecall_match.group(3))  # Recall50 是第三列
    
    # 4. mAP (优先 Known 的 AP50，否则 Prev 的 AP50)
    map_val = None
    known_pattern = r'\|\s*Known\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
    known_match = re.search(known_pattern, content)
    if known_match:
        map_val = float(known_match.group(1))
    else:
        prev_pattern = r'\|\s*Prev\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
        prev_match = re.search(prev_pattern, content)
        if prev_match:
            map_val = float(prev_match.group(1))
    
    return {
        'A-OSE': aose,
        'WI': wi,
        'U-Recall': urecall,
        'mAP': map_val
    }

def collect_data(root, folders):
    """遍历目录，收集数据"""
    # raw_data[stage][alpha][temp] = {指标: 值}
    data = defaultdict(lambda: defaultdict(dict))
    
    for folder_name in folders:
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            print(f"警告：{folder_path} 不存在，跳过")
            continue
        
        # 遍历 alpha 文件夹（数字命名）
        for alpha_name in os.listdir(folder_path):
            alpha_path = os.path.join(folder_path, alpha_name)
            if not os.path.isdir(alpha_path):
                continue
            try:
                alpha = float(alpha_name)  # 确保是数字
            except ValueError:
                continue
            
            # 遍历 temp 文件夹（数字命名）
            for temp_name in os.listdir(alpha_path):
                temp_path = os.path.join(alpha_path, temp_name)
                if not os.path.isdir(temp_path):
                    continue
                try:
                    temp = float(temp_name)
                except ValueError:
                    continue
                
                log_file = os.path.join(temp_path, 'log.txt')
                if not os.path.isfile(log_file):
                    continue
                
                metrics = parse_log(log_file)
                if folder_name == FINAL_STAGE_ONLY_MAP:
                    metrics['A-OSE'] = None
                    metrics['WI'] = None
                    metrics['U-Recall'] = None
                data[folder_name][alpha][temp] = metrics
    
    return data

def plot_metrics(data):
    """按 alpha 分组绘图：每个 alpha 下按 temp 画曲线，横轴为 stage"""
    metrics_names = ['A-OSE', 'WI', 'U-Recall', 'mAP']
    # 科研风格颜色（鲜艳）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 使用用户配置的 stage 顺序，若有额外 stage 再追加
    stage_order = [s for s in FOLDER_LIST if s in data]
    stage_order.extend([s for s in sorted(data.keys()) if s not in stage_order])

    # 汇总所有 (alpha, temp) 组合
    all_pairs = set()
    for stage in stage_order:
        for alpha, temp_dict in data[stage].items():
            for temp in temp_dict.keys():
                all_pairs.add((alpha, temp))
    sorted_pairs = sorted(all_pairs)

    if not sorted_pairs:
        print("未找到任何(alpha, temp)组合，跳过绘图。")
        return

    all_alphas = sorted({alpha for alpha, _ in sorted_pairs})
    for alpha in all_alphas:
        alpha_output_dir = os.path.join(OUTPUT_DIR, f'alpha_{alpha:g}')
        os.makedirs(alpha_output_dir, exist_ok=True)

        temps_for_alpha = sorted({temp for a, temp in sorted_pairs if a == alpha})

        for metric in metrics_names:
            plt.figure(figsize=(8, 5))

            plotted_lines = 0
            for idx, temp in enumerate(temps_for_alpha):
                x = list(range(len(stage_order)))
                y = []
                has_valid = False
                for stage in stage_order:
                    metric_val = data[stage].get(alpha, {}).get(temp, {}).get(metric)
                    if metric_val is None:
                        y.append(float('nan'))
                    else:
                        y.append(metric_val)
                        has_valid = True

                # 该参数组在所有 stage 都缺失该指标时，不绘制
                if not has_valid:
                    continue

                color = colors[idx % len(colors)]
                plt.plot(x, y, marker='o', linestyle='-',
                         linewidth=1.8, markersize=5, color=color,
                         label=f'temp={temp}')
                plotted_lines += 1

            if plotted_lines == 0:
                plt.close()
                continue

            plt.xticks(list(range(len(stage_order))), stage_order)
            plt.xlabel('Stage', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.title(f'{metric} vs Stage | α={alpha:g}', fontsize=14, weight='bold')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True, shadow=True)
            plt.tight_layout()

            output_file_name = f'{metric.replace("-", "").replace(" ", "_")}_alpha_{alpha:g}.svg'
            output_file = os.path.join(alpha_output_dir, output_file_name)
            plt.savefig(output_file, format='svg', bbox_inches='tight')
            print(f"已保存：{output_file} (α={alpha:g}, 共 {plotted_lines} 条 temp 曲线)")
            plt.close()

    print(f"理论参数组数量: {len(sorted_pairs)} (即 alpha-temp 组合数)")

def main():
    print("开始收集数据...")
    data = collect_data(ROOT_DIR, FOLDER_LIST)
    
    if not data:
        print("未找到任何有效数据，请检查路径和文件夹结构。")
        return
    
    # 打印简要统计
    for stage in sorted(data.keys()):
        alpha_dict = data[stage]
        alpha_num = len(alpha_dict)
        temp_num = len({t for temp_dict in alpha_dict.values() for t in temp_dict.keys()})
        print(f"Stage = {stage}: {alpha_num} 个 alpha, {temp_num} 个温度点")
    
    print("\n开始绘制图表...")
    plot_metrics(data)
    print("所有图表绘制完成！")

if __name__ == "__main__":
    main()