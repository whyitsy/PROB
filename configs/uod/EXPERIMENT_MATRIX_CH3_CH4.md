# Chapter 3 / Chapter 4 实验矩阵与消融规划

## 总原则

- `configs/M_OWOD_BENCHMARK.sh`：完整四阶段 baseline 主实验。
- `configs/M_OWOD_BENCHMARK_UOD_CH3.sh`：第三章主实验（显式 unknownness + 稀疏伪监督 + batch 动态分配）。
- `configs/M_OWOD_BENCHMARK_UOD_CH4.sh`：第四章主实验（在 CH3 基础上增加 orth + decorrelation）。
- **T1 消融优先**：所有模块消融与参数敏感性，先在 T1 跑通，趋势成立后再把每章最优配置放到完整 benchmark。

---

## Chapter 3：显式未知性建模与稀疏伪监督协同学习

### 章节主问题
1. 显式 unknownness 分支本身是否有必要？
2. 稀疏伪监督是否真的提供了额外有效监督？
3. batch 级动态分配是否解决了“伪正样本稀少、监督浪费”的问题？

### 主消融矩阵（建议写入正文主表）

| ID | 配置 | 关键开关 | 主要验证问题 |
|---|---|---|---|
| C3-0 | PROB baseline | `--model_type prob` | 作为第三章对照基线 |
| C3-1 | + Explicit Unknownness | `--model_type uod --uod_enable_unknown` | 显式 unknownness 分支本身是否有效 |
| C3-2 | + Sparse Pseudo Supervision | `C3-1 + --uod_enable_pseudo` | 稀疏伪监督是否提供有效弱监督 |
| C3-3 | + Batch Dynamic Allocation | `C3-2 + --uod_enable_batch_dynamic` | 动态分配是否优于仅逐图静态伪监督 |

### 参数敏感性矩阵（建议写入正文小表/附录）

| ID | 固定基础 | 改动参数 | 值 | 主要验证问题 |
|---|---|---|---|---|
| C3-P2 | C3-3 | `uod_batch_topk_ratio` | 0.15 / 0.25 / 0.35 | batch 动态分配强度 |
| C3-P3 | C3-3 | `uod_start_epoch` | 4 / 8 / 12 | 稀疏伪监督启用时机 |

### 第三章重点观察指标
- 主指标：`U_R50 ↑`, `WI ↓`, `A-OSE ↓`
- 稳定性：`K_AP50` 不明显下降
- 过程统计：`stat_num_dummy_pos`, `stat_num_valid_unmatched`, `stat_num_batch_selected_pos`, `stat_pos_thresh_mean`
- 可视化：qualitative overlay、hist_probabilities、scatter_relationships

---

## Chapter 4：objectness–unknownness–classification 解耦去相关优化

### 章节主问题
1. 表示层正交是否有效？
2. 预测层去相关是否有效？
3. 两者是互补还是冗余？

### 主消融矩阵（建议写入正文主表）

| ID | 配置 | 关键开关 | 主要验证问题 |
|---|---|---|---|
| C4-0 | Chapter 3 best | `uod_enable_unknown + uod_enable_pseudo + uod_enable_batch_dynamic` | 作为第四章输入基线 |
| C4-1 | + Orth only | `C4-0 + uod_enable_decorr + uod_orth_loss_coef>0 + uod_decorr_loss_coef=0` | 表示层正交单独贡献 |
| C4-2 | + Decorr only | `C4-0 + uod_enable_decorr + uod_orth_loss_coef=0 + uod_decorr_loss_coef>0` | 预测去相关单独贡献 |
| C4-3 | + Orth + Decorr | `C4-0 + uod_enable_decorr + 两项损失都>0` | 两类解耦是否互补 |

### 参数敏感性矩阵（建议写入正文小表/附录）

| ID | 固定基础 | 改动参数 | 值 | 主要验证问题 |
|---|---|---|---|---|
| C4-P2 | C4-3 | `uod_decorr_loss_coef` | 0.02 / 0.05 / 0.10 | 预测去相关强度 |
| C4-P3 | C4-3 | `uod_orth_loss_coef : uod_decorr_loss_coef` | 1:0 / 0:1 / 1:1 | 两类解耦的主导关系 |

### 第四章重点观察指标
- 主指标：`WI ↓`, `A-OSE ↓`, `U_R50 ↑`
- 稳定性：`K_AP50` 保持或小幅提升
- 结构解释：`correlation_heatmap`, `feature_pca`, `feature_tsne`, `scatter_relationships`

---

## 推荐执行顺序

1. 先跑 `configs/ABLATION_T1_UOD.sh`：看 U0→U3 总趋势是否成立。  
2. 再跑 `configs/ABLATION_T1_UOD_CH3_CORE.sh`：验证第三章机制链。  
3. 再跑 `configs/ABLATION_T1_UOD_CH4_CORE.sh`：验证第四章机制链。  
4. 若主趋势成立，再跑 `configs/ABLATION_T1_UOD_CH3_PARAMS.sh` 和 `configs/ABLATION_T1_UOD_CH4_PARAMS.sh`。  
5. 最后只把 CH3-best 与 CH4-best 放进完整 benchmark 主脚本。  
