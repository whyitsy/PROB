from copy import deepcopy

DEFAULT_VIZ_CFG = {
    # =========================
    # 导出格式
    # =========================
    'figure_format': 'svg',          # 统计图导出格式
    'image_format': 'png',           # 定性图导出格式

    # =========================
    # 采样预算
    # =========================
    'max_qualitative_cases': 12,     # 每次 eval 最多保存多少个定性案例
    'max_tensorboard_cases': 4,      # 最多写入多少个案例到 TensorBoard
    'max_query_samples': 2500,       # 用于统计图的 query 样本上限
    'max_feature_samples': 2500,     # 用于特征嵌入图的样本上限

    # =========================
    # 定性分析开关
    # =========================
    'error_match_iou': 0.50,          # 错误案例匹配时，GT 与预测的最小 IoU
    'save_mining_stage_panel': False, # eval 主链路默认关闭：当前没有稳定 debug provider
    'save_error_panel': True,         # 是否保存 known<->unknown 错误图
    'save_contact_sheet': True,       # 是否保存 contact sheet 汇总图

    # =========================
    # 显示级预测过滤（仅影响可视化）
    # 不影响正式评估指标
    # =========================
    'display_known_score_thresh': 0.20,   # known 面板显示阈值
    'display_unknown_score_thresh': 5.0,  # unknown 面板显示阈值
    'display_nms_iou': 0.30,              # 面板显示用 NMS 阈值
    'display_apply_geometry_filter': True,# 是否启用几何过滤
    'display_min_area_ratio': 0.001,      # 最小面积比例（相对图像面积）
    'display_min_side_ratio': 0.03,       # 最短边最小比例
    'display_max_aspect_ratio': 5.0,      # 最大长宽比

    # =========================
    # 绘图样式
    # =========================
    'min_line_width': 2,
    'line_width_scale': 0.0045,
    'min_font_size': 8,
    'font_size_scale': 0.024,
    'legend_font_size_scale': 0.020,
    'title_font_size_scale': 0.020,
    'info_font_size_scale': 0.019,
    'header_height_ratio': 0.24,
    'header_min_height': 180,
    'header_section_gap': 18,
    'header_text_line_gap': 7,
    'header_box_min_width_ratio': 0.28,
    'header_legend_width_ratio': 0.24,
    'panel_tile_width': 420,
    'panel_tile_height': 280,
    'panel_cols': 2,

    # =========================
    # 统计导出
    # =========================
    'save_query_stats_csv': True,
    'save_feature_npz': True,
    'save_error_summary_csv': True,
    'save_query_distribution_plots': True,
    'save_feature_embedding_plots': False, # 训练期 eval 默认不在线画 embedding，只存离线数据

    # =========================
    # Embedding / manifold 配置
    # =========================
    'embedding_methods': ['pca', 'tsne', 'umap'],
    'embedding_dims': [2, 3],
    'embedding_min_points_2d': 8,
    'embedding_min_points_3d': 12,
    'embedding_tsne_max_points_2d': 800,
    'embedding_tsne_max_points_3d': 500,
    'embedding_umap_max_points_2d': 2000,
    'embedding_umap_max_points_3d': 1200,
    'embedding_generic_max_points_2d': 2000,
    'embedding_generic_max_points_3d': 1200,
    'embedding_tsne_perplexity_cap': 30,
    'embedding_umap_n_neighbors': 20,
    'embedding_umap_min_dist': 0.15,
    'embedding_random_state': 42,
    'save_embedding_group01_views': True,
    'save_embedding_group012_views': True,
    'save_embedding_semantic_views': True,
}


def build_viz_cfg(viz_enabled: bool):
    if not viz_enabled:
        return None
    return deepcopy(DEFAULT_VIZ_CFG)
