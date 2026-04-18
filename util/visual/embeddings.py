import csv
import traceback
import importlib
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from util.visual.helper import save_svg_figure


GROUP_NAMES = ['matched-known', 'unmatched-high-unknown', 'other-unmatched']
GROUP_COLORS = ['#00A65A', '#D81B60', '#6C757D']
SEMANTIC_NAMES = ['matched-known-gt', 'matched-unknown-gt']
SEMANTIC_COLORS = ['#00A65A', '#F39C12']
VIEW_DISPLAY_NAMES = {
    'group01': 'Known / high-unknown focus',
    'group012': 'All queries',
    'semantic_known_unknown': 'Matched GT: known vs unknown',
}


def load_query_statistics(csv_path):
    """读取 query_statistics.csv。"""
    objectness_probability = []
    unknown_probability = []
    max_known_class_probability = []
    query_group = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            objectness_probability.append(float(row['objectness_probability']))
            unknown_probability.append(float(row['unknown_probability']))
            max_known_class_probability.append(float(row['max_known_class_probability']))
            query_group.append(int(row['query_group']))
    return {
        'objectness_probability': np.asarray(objectness_probability, dtype=np.float32),
        'unknown_probability': np.asarray(unknown_probability, dtype=np.float32),
        'max_known_class_probability': np.asarray(max_known_class_probability, dtype=np.float32),
        'query_group': np.asarray(query_group, dtype=np.int64),
    }


def subsample(features, groups, max_points):
    """等间隔下采样点云。"""
    if features.shape[0] <= max_points:
        return features, groups
    indices = np.linspace(0, features.shape[0] - 1, max_points).astype(np.int64)
    return features[indices], groups[indices]


def plot_feature_manifold_3d(npz_path, output_path, max_points=1500):
    """绘制三组 feature 的 3D manifold。"""
    data = np.load(npz_path)
    groups = data['feature_groups'].astype(np.int64)
    feature_specs = [
        ('objectness_features', 'Objectness feature manifold'),
        ('knownness_features', 'Knownness feature manifold'),
        ('classification_features', 'Classification feature manifold'),
    ]
    fig = plt.figure(figsize=(18.5, 6.0), constrained_layout=True)
    for subplot_index, (key, title) in enumerate(feature_specs, start=1):
        if key not in data:
            continue
        features = data[key].astype(np.float32)
        features, feature_groups = subsample(features, groups, max_points)
        projection = PCA(n_components=3, random_state=42).fit_transform(features)
        ax = fig.add_subplot(1, 3, subplot_index, projection='3d')
        for group_index, (group_name, color) in enumerate(zip(GROUP_NAMES, GROUP_COLORS)):
            mask = feature_groups == group_index
            if np.any(mask):
                ax.scatter(projection[mask, 0], projection[mask, 1], projection[mask, 2], s=10, alpha=0.62, c=color, label=group_name)
        ax.set_title(title)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
    if fig.axes:
        handles, labels = fig.axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    return save_svg_figure(fig, output_path)


def plot_score_space_3d(stats, output_path, max_points=1500):
    """绘制 3D score space 散点图。"""
    features = np.stack(
        [
            stats['objectness_probability'],
            stats['unknown_probability'],
            stats['max_known_class_probability'],
        ],
        axis=1,
    )
    features, groups = subsample(features, stats['query_group'], max_points)
    fig = plt.figure(figsize=(8.4, 6.8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for group_index, (group_name, color) in enumerate(zip(GROUP_NAMES, GROUP_COLORS)):
        mask = groups == group_index
        if np.any(mask):
            ax.scatter(features[mask, 0], features[mask, 1], features[mask, 2], s=10, alpha=0.62, c=color, label=group_name)
    ax.set_xlabel('objectness prob')
    ax.set_ylabel('unknown prob')
    ax.set_zlabel('max known prob')
    ax.set_title('Score-space decision geometry (3D)')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    return save_svg_figure(fig, output_path)


def plot_score_space_slices(stats, output_path, max_points=1500):
    """绘制 score space 的 2D 切片图。"""
    features = np.stack(
        [
            stats['objectness_probability'],
            stats['unknown_probability'],
            stats['max_known_class_probability'],
        ],
        axis=1,
    )
    features, groups = subsample(features, stats['query_group'], max_points)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
    slice_specs = [
        ((0, 1), 'objectness prob', 'unknown prob', 'Objectness vs Unknownness'),
        ((0, 2), 'objectness prob', 'max known prob', 'Objectness vs Max Known'),
        ((1, 2), 'unknown prob', 'max known prob', 'Unknownness vs Max Known'),
    ]
    for ax, ((x_idx, y_idx), x_label, y_label, title) in zip(axes, slice_specs):
        for group_index, (group_name, color) in enumerate(zip(GROUP_NAMES, GROUP_COLORS)):
            mask = groups == group_index
            if np.any(mask):
                ax.scatter(features[mask, x_idx], features[mask, y_idx], s=10, alpha=0.62, c=color, label=group_name)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    return save_svg_figure(fig, output_path)


def select_embedding_max_points(method, dim, viz_cfg):
    """选择不同 embedding 的采样上限。"""
    if method == 'tsne':
        return int(viz_cfg['embedding_tsne_max_points_2d'] if dim == 2 else viz_cfg['embedding_tsne_max_points_3d'])
    if method == 'umap':
        return int(viz_cfg['embedding_umap_max_points_2d'] if dim == 2 else viz_cfg['embedding_umap_max_points_3d'])
    return int(viz_cfg['embedding_generic_max_points_2d'] if dim == 2 else viz_cfg['embedding_generic_max_points_3d'])


def subsample_evenly(features, labels, max_points):
    """等间隔下采样特征。"""
    if features.shape[0] <= max_points:
        return features, labels
    indices = np.linspace(0, features.shape[0] - 1, max_points).astype(np.int64)
    return features[indices], labels[indices]


def compute_embedding(features, method, dim, viz_cfg):
    """计算降维嵌入。"""
    random_state = int(viz_cfg.get('embedding_random_state', 42))
    if method == 'pca':
        return PCA(n_components=dim, random_state=random_state).fit_transform(features)
    if method == 'tsne':
        perplexity = min(int(viz_cfg.get('embedding_tsne_perplexity_cap', 30)), max(2, features.shape[0] // 4))
        return TSNE(
            n_components=dim,
            perplexity=perplexity,
            init='pca',
            learning_rate='auto',
            random_state=random_state,
        ).fit_transform(features)
    if method == 'umap':
        spec_root = importlib.util.find_spec('umap')
        spec_sub = importlib.util.find_spec('umap.umap_')
        if spec_root is None:
            raise ImportError('Cannot find package "umap"')
        if spec_sub is None:
            raise ImportError(f'Cannot find submodule "umap.umap_" (root origin={getattr(spec_root, "origin", None)})')
        umap_module = importlib.import_module('umap.umap_')
        UMAP = getattr(umap_module, 'UMAP')
        n_neighbors = min(int(viz_cfg.get('embedding_umap_n_neighbors', 20)), max(2, features.shape[0] - 1))
        reducer = UMAP(
            n_components=dim,
            n_neighbors=n_neighbors,
            min_dist=float(viz_cfg.get('embedding_umap_min_dist', 0.15)),
            random_state=random_state,
        )
        return reducer.fit_transform(features)
    raise ValueError(f'Unsupported embedding method: {method}')


def scatter_embedding(ax, embedding, labels, names, colors, dim):
    """把 embedding 画到坐标轴上。"""
    for group_index, (name, color) in enumerate(zip(names, colors)):
        mask = labels == group_index
        if not np.any(mask):
            continue
        if dim == 3:
            ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2], s=9, alpha=0.58, c=color, label=name)
            ax.set_zlabel('Component 3')
        else:
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=9, alpha=0.58, c=color, label=name)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(alpha=0.2)


def create_embedding_axes(dim, count):
    """创建 embedding 图的坐标轴。"""
    if dim == 3:
        fig = plt.figure(figsize=(17.8, 5.6))
        axes = [fig.add_subplot(1, count, index + 1, projection='3d') for index in range(count)]
    else:
        fig, axes = plt.subplots(1, count, figsize=(16.4, 5.1))
        axes = list(axes)
    return fig, axes


def build_view_specs(query_groups, feature_is_matched, feature_matched_gt_is_unknown):
    """构建 embedding 视图配置。"""
    return [
        {
            'key': 'group01',
            'display_name': VIEW_DISPLAY_NAMES['group01'],
            'mask': np.isin(query_groups, [0, 1]),
            'labels': np.where(query_groups == 0, 0, 1),
            'names': ['matched-known', 'unmatched-high-unknown'],
            'colors': [GROUP_COLORS[0], GROUP_COLORS[1]],
        },
        {
            'key': 'group012',
            'display_name': VIEW_DISPLAY_NAMES['group012'],
            'mask': np.ones_like(query_groups, dtype=bool),
            'labels': query_groups,
            'names': GROUP_NAMES,
            'colors': GROUP_COLORS,
        },
        {
            'key': 'semantic_known_unknown',
            'display_name': VIEW_DISPLAY_NAMES['semantic_known_unknown'],
            'mask': feature_is_matched.astype(bool),
            'labels': feature_matched_gt_is_unknown.astype(np.int64),
            'names': SEMANTIC_NAMES,
            'colors': SEMANTIC_COLORS,
        },
    ]


def embedding_output_dir(output_dirs, family, dim, method):
    """返回 embedding 输出目录。"""
    base_key = f'{family}_{dim}d'
    return Path(output_dirs[base_key]) / method


def write_embedding_error(output_dir, filename_stem, error, trace_text=None, extra_info=None):
    """写 embedding 失败日志。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_path = output_dir / f'{filename_stem}_error.txt'
    with error_path.open('w', encoding='utf-8') as file:
        file.write(f'ERROR: {repr(error)}\\n\\n')
        if extra_info:
            file.write('EXTRA INFO:\\n')
            for key, value in extra_info.items():
                file.write(f'- {key}: {value}\\n')
            file.write('\\n')
        if trace_text:
            file.write('TRACEBACK:\\n')
            file.write(trace_text)


def plot_feature_embedding_views(state, output_dirs, viz_cfg):
    """批量绘制 feature embedding 视图。"""
    if not state['feature_groups'] or not state['objectness_features']:
        return
    feature_specs = [
        ('objectness_features', 'Objectness'),
        ('knownness_features', 'Knownness'),
        ('classification_features', 'Classification'),
    ]
    query_groups = np.asarray(state['feature_groups'], dtype=np.int64)
    feature_is_matched = np.asarray(state['feature_is_matched'], dtype=np.int64)
    feature_matched_gt_is_unknown = np.asarray(state['feature_matched_gt_is_unknown'], dtype=np.int64)

    for dim in viz_cfg.get('embedding_dims', [2, 3]):
        min_points = int(viz_cfg['embedding_min_points_2d'] if dim == 2 else viz_cfg['embedding_min_points_3d'])
        for method in viz_cfg.get('embedding_methods', ['pca', 'tsne', 'umap']):
            for view in build_view_specs(query_groups, feature_is_matched, feature_matched_gt_is_unknown):
                output_dir = embedding_output_dir(output_dirs, 'feature', dim, method)
                fig, axes = create_embedding_axes(dim, len(feature_specs))
                plotted_any = False
                legend_handles = None
                legend_labels = None
                filename_stem = f'feature_embedding_{method}_{dim}d_{view["key"]}'
                for ax, (feature_key, feature_title) in zip(axes, feature_specs):
                    features = np.asarray(state[feature_key], dtype=np.float32)
                    labels = view['labels'][view['mask']]
                    masked_features = features[view['mask']]
                    if masked_features.shape[0] < min_points or np.unique(labels).size < 2:
                        ax.set_axis_off()
                        continue
                    masked_features, labels = subsample_evenly(
                        masked_features,
                        labels,
                        select_embedding_max_points(method, dim, viz_cfg),
                    )
                    try:
                        embedding = compute_embedding(masked_features, method, dim, viz_cfg)
                    except Exception as error:
                        ax.set_axis_off()
                        write_embedding_error(
                            output_dir,
                            filename_stem,
                            error,
                            trace_text=traceback.format_exc(),
                            extra_info={
                                'method': method,
                                'dim': dim,
                                'view_key': view['key'],
                                'umap_root_origin': getattr(importlib.util.find_spec('umap'), 'origin', None),
                                'umap_sub_origin': getattr(importlib.util.find_spec('umap.umap_'), 'origin', None),
                            },
                        )
                        continue
                    scatter_embedding(ax, embedding, labels, view['names'], view['colors'], dim)
                    ax.set_title(feature_title)
                    handles, labels_text = ax.get_legend_handles_labels()
                    if handles:
                        legend_handles, legend_labels = handles, labels_text
                    plotted_any = True
                if plotted_any:
                    if legend_handles:
                        fig.legend(
                            legend_handles,
                            legend_labels,
                            loc='upper center',
                            bbox_to_anchor=(0.5, 1.01),
                            ncol=max(2, len(legend_labels)),
                            frameon=False,
                        )
                    fig.suptitle(f'{method.upper()} · {view["display_name"]}', y=1.04 if dim == 2 else 1.02)
                    save_svg_figure(fig, output_dir / f'{filename_stem}.{viz_cfg["figure_format"]}')
                else:
                    plt.close(fig)


def plot_score_space_embedding_views(state, output_dirs, viz_cfg):
    """批量绘制 score space embedding 视图。"""
    if not state['objectness_probability']:
        return
    features = np.stack(
        [
            np.asarray(state['objectness_probability'], dtype=np.float32),
            np.asarray(state['unknown_probability'], dtype=np.float32),
            np.asarray(state['max_known_class_probability'], dtype=np.float32),
        ],
        axis=1,
    )
    query_groups = np.asarray(state['query_group'], dtype=np.int64)
    feature_is_matched = np.asarray(state['is_matched'], dtype=np.int64)
    feature_matched_gt_is_unknown = np.asarray(state['matched_gt_is_unknown'], dtype=np.int64)

    for dim in viz_cfg.get('embedding_dims', [2, 3]):
        min_points = int(viz_cfg['embedding_min_points_2d'] if dim == 2 else viz_cfg['embedding_min_points_3d'])
        for method in viz_cfg.get('embedding_methods', ['pca', 'tsne', 'umap']):
            for view in build_view_specs(query_groups, feature_is_matched, feature_matched_gt_is_unknown):
                output_dir = embedding_output_dir(output_dirs, 'score', dim, method)
                masked_features = features[view['mask']]
                labels = view['labels'][view['mask']]
                filename_stem = f'score_space_{method}_{dim}d_{view["key"]}'
                if masked_features.shape[0] < min_points or np.unique(labels).size < 2:
                    continue
                masked_features, labels = subsample_evenly(
                    masked_features,
                    labels,
                    select_embedding_max_points(method, dim, viz_cfg),
                )
                try:
                    embedding = compute_embedding(masked_features, method, dim, viz_cfg)
                except Exception as error:
                    write_embedding_error(
                        output_dir,
                        filename_stem,
                        error,
                        trace_text=traceback.format_exc(),
                        extra_info={
                            'method': method,
                            'dim': dim,
                            'view_key': view['key'],
                            'umap_root_origin': getattr(importlib.util.find_spec('umap'), 'origin', None),
                            'umap_sub_origin': getattr(importlib.util.find_spec('umap.umap_'), 'origin', None),
                        },
                    )
                    continue
                if dim == 3:
                    fig = plt.figure(figsize=(7.8, 6.2))
                    ax = fig.add_subplot(1, 1, 1, projection='3d')
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.8))
                scatter_embedding(ax, embedding, labels, view['names'], view['colors'], dim)
                ax.set_title('Score-space embedding')
                ax.legend(
                    frameon=False,
                    fontsize=8,
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1.10),
                    ncol=max(2, len(view['names'])),
                )
                fig.suptitle(f'{method.upper()} · {view["display_name"]}', y=1.02)
                save_svg_figure(fig, output_dir / f'{filename_stem}.{viz_cfg["figure_format"]}')
