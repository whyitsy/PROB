import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

VIEW_SPECS = {
    'group01': {
        'display_name': 'Known / high-unknown focus',
        'axis_prefix': 'Known / high-unknown',
        'legend_names': ['Known', 'High-unknown'],
        'legend_colors': ['#00A65A', '#D81B60'],
    },
    'group012': {
        'display_name': 'All queries',
        'axis_prefix': 'Query-group',
        'legend_names': ['Known', 'High-unknown', 'Other unmatched'],
        'legend_colors': ['#00A65A', '#D81B60', '#6C757D'],
    },
    'semantic_known_unknown': {
        'display_name': 'Matched GT: known vs unknown',
        'axis_prefix': 'Known / unknown',
        'legend_names': ['GT known', 'GT unknown'],
        'legend_colors': ['#00A65A', '#F39C12'],
    },
}

FEATURE_SPECS = [
    ('objectness_features', 'Objectness'),
    ('knownness_features', 'Knownness'),
    ('classification_features', 'Classification'),
]


def load_query_statistics(csv_path):
    result = {
        'objectness_probability': [],
        'unknown_probability': [],
        'max_known_class_probability': [],
        'query_group': [],
        'is_matched': [],
        'matched_gt_is_unknown': [],
    }
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            result['objectness_probability'].append(float(row['objectness_probability']))
            result['unknown_probability'].append(float(row['unknown_probability']))
            result['max_known_class_probability'].append(float(row['max_known_class_probability']))
            result['query_group'].append(int(float(row['query_group'])))
            result['is_matched'].append(int(float(row['is_matched'])))
            result['matched_gt_is_unknown'].append(int(float(row['matched_gt_is_unknown'])))
    for key, values in result.items():
        dtype = np.float32 if 'probability' in key else np.int64
        result[key] = np.asarray(values, dtype=dtype)
    return result


def load_feature_samples(npz_path):
    return {key: value for key, value in np.load(npz_path).items()}


def subsample(features, labels, max_points):
    if features.shape[0] <= max_points:
        return features, labels
    indices = np.linspace(0, features.shape[0] - 1, max_points).astype(np.int64)
    return features[indices], labels[indices]


def compute_embedding(features, method, dim, random_state=42):
    if method == 'pca':
        return PCA(n_components=dim, random_state=random_state).fit_transform(features)
    if method == 'tsne':
        perplexity = min(30, max(2, features.shape[0] // 4))
        return TSNE(n_components=dim, perplexity=perplexity, init='pca', learning_rate='auto', random_state=random_state).fit_transform(features)
    if method == 'umap':
        from umap import UMAP
        n_neighbors = min(20, max(2, features.shape[0] - 1))
        reducer = UMAP(n_components=dim, n_neighbors=n_neighbors, min_dist=0.15, random_state=random_state)
        return reducer.fit_transform(features)
    raise ValueError(f'Unsupported method: {method}')


def scatter_embedding(axis, embedding, labels, view_key, dim):
    spec = VIEW_SPECS[view_key]
    for group_index, (group_name, color) in enumerate(zip(spec['legend_names'], spec['legend_colors'])):
        mask = labels == group_index
        if not np.any(mask):
            continue
        if dim == 3:
            axis.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2], s=8, alpha=0.58, c=color, label=group_name)
            axis.set_zlabel(f"{spec['axis_prefix']} axis 3")
        else:
            axis.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.58, c=color, label=group_name)
        axis.set_xlabel(f"{spec['axis_prefix']} axis 1")
        axis.set_ylabel(f"{spec['axis_prefix']} axis 2")
        axis.grid(alpha=0.2)


def create_axes(dim, count, figsize):
    if dim == 3:
        figure = plt.figure(figsize=figsize)
        axes = [figure.add_subplot(1, count, idx + 1, projection='3d') for idx in range(count)]
    else:
        figure, axes = plt.subplots(1, count, figsize=figsize)
        axes = list(axes) if isinstance(axes, np.ndarray) else [axes]
    return figure, axes


def build_view_specs(groups, is_matched, matched_gt_is_unknown):
    return [
        ('group01', np.isin(groups, [0, 1]), np.where(groups == 0, 0, 1).astype(np.int64)),
        ('group012', np.ones_like(groups, dtype=bool), groups.astype(np.int64)),
        ('semantic_known_unknown', is_matched.astype(bool), matched_gt_is_unknown.astype(np.int64)),
    ]


def ensure_output_dirs(stats_dir):
    paths = {}
    for family in ['feature', 'score_space']:
        for dim in ['2d', '3d']:
            for method in ['pca', 'tsne', 'umap']:
                path = stats_dir / 'embeddings' / 'standalone' / family / dim / method
                path.mkdir(parents=True, exist_ok=True)
                paths[f'{family}_{dim}_{method}'] = path
    return paths


def plot_feature_embeddings(npz_data, output_dirs, max_points):
    groups = np.asarray(npz_data['feature_groups'], dtype=np.int64)
    is_matched = np.asarray(npz_data['feature_is_matched'], dtype=np.int64)
    matched_gt_is_unknown = np.asarray(npz_data['feature_matched_gt_is_unknown'], dtype=np.int64)
    for dim in [2, 3]:
        min_points = 8 if dim == 2 else 12
        for method in ['pca', 'tsne', 'umap']:
            method_dir = output_dirs[f'feature_{dim}d_{method}']
            for view_key, mask, labels_all in build_view_specs(groups, is_matched, matched_gt_is_unknown):
                figure, axes = create_axes(dim, 3, (18, 6.4) if dim == 3 else (17.2, 5.8))
                plotted_any = False
                handles = None
                legend_labels = None
                for axis, (key, title) in zip(axes, FEATURE_SPECS):
                    features = np.asarray(npz_data[key], dtype=np.float32)[mask]
                    labels = labels_all[mask]
                    if features.shape[0] < min_points or np.unique(labels).size < 2:
                        axis.set_axis_off()
                        continue
                    features, labels = subsample(features, labels, max_points)
                    embedding = compute_embedding(features, method, dim)
                    scatter_embedding(axis, embedding, labels, view_key, dim)
                    axis.set_title(title, fontsize=11)
                    h, l = axis.get_legend_handles_labels()
                    if h:
                        handles, legend_labels = h, l
                    plotted_any = True
                if not plotted_any:
                    plt.close(figure)
                    continue
                figure.suptitle(f"{method.upper()} · {VIEW_SPECS[view_key]['display_name']}", fontsize=13, y=0.98)
                if handles:
                    figure.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=max(2, len(legend_labels)), frameon=False)
                figure.tight_layout(rect=[0, 0, 1, 0.88])
                figure.savefig(method_dir / f'feature_embedding_{dim}d_{view_key}.png', bbox_inches='tight')
                plt.close(figure)


def plot_score_space(stats, output_dirs, max_points):
    features = np.stack([
        stats['objectness_probability'],
        stats['unknown_probability'],
        stats['max_known_class_probability'],
    ], axis=1).astype(np.float32)
    groups = stats['query_group'].astype(np.int64)
    is_matched = stats['is_matched'].astype(np.int64)
    matched_gt_is_unknown = stats['matched_gt_is_unknown'].astype(np.int64)
    for dim in [2, 3]:
        min_points = 8 if dim == 2 else 12
        for method in ['pca', 'tsne', 'umap']:
            method_dir = output_dirs[f'score_space_{dim}d_{method}']
            for view_key, mask, labels_all in build_view_specs(groups, is_matched, matched_gt_is_unknown):
                masked_features = features[mask]
                labels = labels_all[mask]
                if masked_features.shape[0] < min_points or np.unique(labels).size < 2:
                    continue
                masked_features, labels = subsample(masked_features, labels, max_points)
                embedding = compute_embedding(masked_features, method, dim)
                figure, axes = create_axes(dim, 1, (8.0, 6.8) if dim == 3 else (7.6, 6.2))
                axis = axes[0]
                scatter_embedding(axis, embedding, labels, view_key, dim)
                axis.set_title('Score space', fontsize=11)
                h, l = axis.get_legend_handles_labels()
                if h:
                    figure.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=max(2, len(l)), frameon=False)
                figure.suptitle(f"{method.upper()} · {VIEW_SPECS[view_key]['display_name']}", fontsize=13, y=0.98)
                figure.tight_layout(rect=[0, 0, 1, 0.88])
                figure.savefig(method_dir / f'score_space_{dim}d_{view_key}.png', bbox_inches='tight')
                plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser('Standalone rerender for PCA / t-SNE / UMAP embeddings from stats/data')
    parser.add_argument('--stats_dir', required=True, type=str, help='path like eval/visualizations/epoch_xxxx/stats')
    parser.add_argument('--max_points', default=1200, type=int)
    return parser


def main(args):
    stats_dir = Path(args.stats_dir)
    data_dir = stats_dir / 'data'
    npz_path = data_dir / 'feature_samples.npz'
    csv_path = data_dir / 'query_statistics.csv'
    output_dirs = ensure_output_dirs(stats_dir)
    if npz_path.exists():
        plot_feature_embeddings(load_feature_samples(npz_path), output_dirs, args.max_points)
    if csv_path.exists():
        plot_score_space(load_query_statistics(csv_path), output_dirs, args.max_points)
    print(f'Saved standalone rerender plots under: {stats_dir / "embeddings" / "standalone"}')


if __name__ == '__main__':
    main(build_parser().parse_args())
