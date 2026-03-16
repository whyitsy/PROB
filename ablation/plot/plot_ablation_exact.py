import os
import re
import matplotlib.pyplot as plt

# ==========================================
# 1. 实验路径与名称配置
# ==========================================
# 请确保与你 run_ablation_t1.sh 中的 BASE_EXP_DIR 保持一致
BASE_DIR = "/mnt/data/kky/output/PROB/exps/MOWODB/ABLATION_STUDY"

EXPERIMENTS = [
    "Exp1_Baseline",
    "Exp2_Innov1_HardLabel",
    "Exp3_Innov2_Align",
    "Exp4_Innov2_FullVSAD"
]

LEGEND_NAMES = [
    "Baseline (PROB)",
    "+ ETOP & TDQI & Mask",
    "+ Feature Align",
    "+ Full VSAD (Ours)"
]

# ==========================================
# 2. 针对 open_world_eval.py 的精准正则匹配
# ==========================================
PATTERNS = {
    # 从 "| Unknown | 1.23 | 0.10 | 18.50 |" 提取第四列的 Recall50
    "U-Recall": re.compile(r"\|\s*Unknown\s*\|\s*\d+\.\d+\s*\|\s*\d+\.\d+\s*\|\s*(\d+\.\d+)"),
    
    # 从 "| Known | 45.67 | 34.56 | 67.89 |" 提取第二列的 AP50
    "Known_mAP": re.compile(r"\|\s*Known\s*\|\s*(\d+\.\d+)"),
    
    # 从 "| 0.8000 | 0.0567 | ..." 提取 Recall=0.8 时的 WI-IoU50
    "WI": re.compile(r"\|\s*0\.8000\s*\|\s*(\d+\.\d+)"),
    
    # 提取 "Absolute OSE (IoU=50): 1234"
    "A-OSE": re.compile(r"Absolute OSE \(IoU=50\):\s+(\d+(\.\d+)?)")
}

def extract_metrics_from_log(log_path):
    """逐行解析日志并提取每个 Epoch 的 4 个核心指标"""
    metrics = {k: [] for k in PATTERNS.keys()}
    
    if not os.path.exists(log_path):
        print(f"[警告] 找不到日志文件，将跳过: {log_path}")
        return metrics

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            for metric_name, pattern in PATTERNS.items():
                match = pattern.search(line)
                if match:
                    # 匹配成功，将捕获的数字加入列表
                    metrics[metric_name].append(float(match.group(1)))
                    
    # 打印提取到的数量，用于校验是否存在由于中途崩溃导致的指标数量不齐
    print(f"  -> 成功提取记录数: " + ", ".join([f"{k}:{len(v)}" for k, v in metrics.items()]))
    return metrics

# ==========================================
# 3. 主干提取逻辑
# ==========================================
all_data = {}
for exp in EXPERIMENTS:
    # 适配 main_open_world.py 自动生成的 log.txt 路径
    log_file = os.path.join(BASE_DIR, exp, "t1", "log.txt")
    print(f"正在解析实验: {exp} ...")
    all_data[exp] = extract_metrics_from_log(log_file)

# ==========================================
# 4. 论文级高质量绘图
# ==========================================
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14})

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ablation Study Metrics over Epochs (Task 1)', fontsize=18, fontweight='bold', y=0.96)

metric_keys = list(PATTERNS.keys())
axes_flat = axs.flatten()

# 统一配色：灰, 蓝, 橙, 红 (符合学术论文审美，突出最终版本)
colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#d62728'] 
markers = ['o', 's', '^', 'D']

for i, metric in enumerate(metric_keys):
    ax = axes_flat[i]
    
    for j, exp in enumerate(EXPERIMENTS):
        y_values = all_data[exp][metric]
        if not y_values:
            continue
            
        # 以实际评估次数作为 X 轴
        x_values = list(range(1, len(y_values) + 1))
        
        ax.plot(x_values, y_values, label=LEGEND_NAMES[j], 
                color=colors[j], marker=markers[j], markersize=5, linewidth=2, alpha=0.85)
    
    ax.set_title(metric)
    ax.set_xlabel('Evaluation Steps')
    
    # 设置 Y 轴标签
    if metric == 'A-OSE':
        ax.set_ylabel('Absolute Count')
    elif metric == 'WI':
        ax.set_ylabel('Ratio')
    else:
        ax.set_ylabel('Score (%)')
        
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 根据指标好坏调整图例位置，避免遮挡曲线
    if metric in ["WI", "A-OSE"]:
        ax.legend(loc='upper right') # 这两个指标越低越好
    else:
        ax.legend(loc='lower right') # U-Recall 和 K-mAP 越高越好

plt.tight_layout()
plt.subplots_adjust(top=0.90) # 给主标题留白

# 保存为高清 PNG 供论文使用
save_path = os.path.join(BASE_DIR, "ablation_results_exact.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n[成功] 折线图已成功生成并保存至: {save_path}")