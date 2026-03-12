import re
import matplotlib.pyplot as plt
import os

def parse_log_for_metrics(log_path):
    """
    从训练日志中提取 Known AP50 (K_AP50) 和 Unknown Recall (U_R50)
    """
    k_ap50_list = []
    u_r50_list = []
    
    # 匹配字典打印格式，例如: 'K_AP50': 55.42 或 "K_AP50": 55.42
    # 兼容单引号和双引号，以及可能存在的浮点数
    k_pattern = re.compile(r'[\'"]K_AP50[\'"]\s*:\s*([0-9.]+)')
    u_pattern = re.compile(r'[\'"]U_R50[\'"]\s*:\s*([0-9.]+)')
    
    if not os.path.exists(log_path):
        print(f"警告：找不到日志文件 {log_path}")
        return [], []
        
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'K_AP50' in line and 'U_R50' in line:
                k_match = k_pattern.search(line)
                u_match = u_pattern.search(line)
                
                if k_match and u_match:
                    k_ap50_list.append(float(k_match.group(1)))
                    u_r50_list.append(float(u_match.group(1)))
                    
    return k_ap50_list, u_r50_list

def plot_thesis_charts(log_baseline, log_ours, save_dir="."):
    """
    绘制并保存学术级别的对比折线图
    """
    # 1. 解析数据
    base_k, base_u = parse_log_for_metrics(log_baseline)
    ours_k, ours_u = parse_log_for_metrics(log_ours)
    
    # 对齐 Epoch 长度 (以防两个实验跑的 Epoch 数量不同)
    epochs_base = list(range(1, len(base_k) + 1))
    epochs_ours = list(range(1, len(ours_k) + 1))
    
    # 2. 设置全局绘图风格 (符合毕业论文标准)
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'sans-serif',
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.figsize': (14, 6) # 宽一点，方便并排画两张图
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # --- 图 1: Unknown Recall (U-Recall50) 对比 ---
    ax1.plot(epochs_base, base_u, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Baseline (ETOP+TDQI)')
    ax1.plot(epochs_ours, ours_u, marker='^', linestyle='-', color='#d62728', linewidth=2, label='Ours (+ VSAD & SVCF)')
    
    ax1.set_title('Unknown Recall (U-Recall50) Progression')
    ax1.set_xlabel('Evaluation Steps / Epochs')
    ax1.set_ylabel('U-Recall50 (%)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right')
    
    # --- 图 2: Known mAP (K-AP50) 对比 ---
    ax2.plot(epochs_base, base_k, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Baseline (ETOP+TDQI)')
    ax2.plot(epochs_ours, ours_k, marker='^', linestyle='-', color='#d62728', linewidth=2, label='Ours (+ VSAD & SVCF)')
    
    ax2.set_title('Known mAP (K-AP50) Progression')
    ax2.set_xlabel('Evaluation Steps / Epochs')
    ax2.set_ylabel('K-AP50 (%)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right')
    
    # 3. 调整布局并保存为高分辨率图片 (支持 PDF 矢量图或 PNG)
    plt.tight_layout()
    
    png_path = os.path.join(save_dir, 'metrics_comparison.png')
    pdf_path = os.path.join(save_dir, 'metrics_comparison.pdf') # 强烈建议在论文中使用 PDF 矢量图格式
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    print(f"图表已成功生成并保存至:\n - {png_path}\n - {pdf_path}")

if __name__ == "__main__":
    # 请在这里替换为你实际的 log 文件路径
    # log_stage1 对应你纯视觉挖掘的实验
    # log_stage2 对应你加入了 CLIP 对齐和 SVCF 的实验
    LOG_STAGE1_PATH = "logs/train_stage1.txt" 
    LOG_STAGE2_PATH = "logs/train_stage2.txt"
    
    plot_thesis_charts(LOG_STAGE1_PATH, LOG_STAGE2_PATH)