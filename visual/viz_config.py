from copy import deepcopy

DEFAULT_VIZ_CFG = {
    # Export policy
    'figure_format': 'svg',
    'image_format': 'png',
    # Sampling budget
    'max_qualitative_cases': 12,
    'max_tensorboard_cases': 4,
    'max_query_samples': 2500,
    'max_feature_samples': 2500,
    # Qualitative analysis
    'error_match_iou': 0.50,
    'save_mining_stage_panel': True,
    'save_error_panel': True,
    'save_contact_sheet': True,
    # Rendering style (visual-only, does not alter model semantics)
    'min_line_width': 2,
    'line_width_scale': 0.0045,
    'min_font_size': 12,
    'font_size_scale': 0.028,
    'legend_font_size_scale': 0.022,
    'panel_tile_width': 420,
    'panel_tile_height': 280,
    'panel_cols': 2,
    # Statistics export
    'save_query_stats_csv': True,
    'save_feature_npz': True,
    'save_error_summary_csv': True,
    'save_query_distribution_plots': True,
    'save_feature_embedding_plots': True,
    # Display filtering
    'display_known_score_thresh': 0.25,
    'display_unknown_score_thresh': 0.25,
    'display_apply_geometry_filter': False,
    'display_min_area_ratio': 0.0001,
    'display_min_side_ratio': 0.01,
    'display_max_aspect_ratio': 20.0,
    'display_nms_iou': 0.60,
    # Embedding controls
    'embedding_dims': [2, 3],
    'embedding_methods': ['pca', 'tsne'],
    'embedding_min_points_2d': 8,
    'embedding_min_points_3d': 12,
    'embedding_generic_max_points_2d': 2000,
    'embedding_generic_max_points_3d': 1500,
    'embedding_tsne_max_points_2d': 800,
    'embedding_tsne_max_points_3d': 600,
    'embedding_umap_max_points_2d': 1200,
    'embedding_umap_max_points_3d': 900,
    'embedding_random_state': 42,
    'embedding_tsne_perplexity_cap': 30,
    'embedding_umap_n_neighbors': 20,
    'embedding_umap_min_dist': 0.15,
}


def build_viz_cfg(viz_enabled: bool):
    if not viz_enabled:
        return None
    return deepcopy(DEFAULT_VIZ_CFG)
