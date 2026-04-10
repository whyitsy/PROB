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
}


def build_viz_cfg(viz_enabled: bool):
    if not viz_enabled:
        return None
    return deepcopy(DEFAULT_VIZ_CFG)
