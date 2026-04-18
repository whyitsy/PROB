import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

GROUP_NAMES = ['matched-known', 'unmatched-high-unknown', 'other-unmatched']
GROUP_COLORS = ['#00A65A', '#D81B60', '#6C757D']
SEMANTIC_NAMES = ['matched-known-gt', 'matched-unknown-gt']
SEMANTIC_COLORS = ['#00A65A', '#F39C12']


def load_query_statistics(csv_path):
    result = {
        'objectness_probability': [],
        'unknown_probability': [],
        'max_known_class_probability': [],
        'query_group': [],
        'is_matched': [],
        'matched_gt_label': [],
        'matched_gt_is_unknown': [],
        'pred_top1_label': [],
        'pred_top1_is_unknown': [],
        'top1_known_class': [],
        'image_id': [],
        'query_index': [],
    }
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key in result.keys():
                value = row.get(key, None)
                if value is None or value == '':
                    value = -1
                result[key].append(float(value) if 'probability' in key else int(float(value)))
    for key, values in result.items():
        dtype = np.float32 if 'probability' in key else np.int64
        result[key] = np.asarray(values, dtype=dtype)
    return result


def load_feature_samples(npz_path):
    data = np.load(npz_path)
    result = {key: data[key] for key in data.files}
    return result


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
        return TSNE(
            n_components=dim,
            perplexity=perplexity,
            init='pca',
            learning_rate='auto',
            random_state=random_state,
        ).fit_transform(features)
    if method == 'umap':
        n_neighbors = min(20, max(2, features.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=dim,
            n_neighbors=n_neighbors,
            min_dist=0.15,
            random_state=random_state,
        )
        return reducer.fit_transform(features)
    raise ValueError(f'Unsupported method: {method}')


def scatter_embedding(axis, embedding, labels, names, colors, dim):
    for group_index, (group_name, color) in enumerate(zip(names, colors)):
        mask = labels == group_index
        if not np.any(mask):
            continue
        if dim == 3:
            axis.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2], s=8, alpha=0.58, c=color, label=group_name)
            axis.set_zlabel('C3')
        else:
            axis.scatter(embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.58, c=color, label=group_name)
        axis.set_xlabel('C1')
        axis.set_ylabel('C2')
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
        ('group01', np.isin(groups, [0, 1]), np.where(groups == 0, 0, 1), ['matched-known', 'unmatched-high-unknown'], ['#00A65A', '#D81B60']),
        ('group012', np.ones_like(groups, dtype=bool), groups, GROUP_NAMES, GROUP_COLORS),
        ('semantic_known_unknown', is_matched.astype(bool), matched_gt_is_unknown.astype(np.int64), SEMANTIC_NAMES, SEMANTIC_COLORS),
    ]


def plot_feature_embeddings(npz_data, output_dir, methods, dims, max_points):
    groups = np.asarray(npz_data['feature_groups'], dtype=np.int64)
    is_matched = np.asarray(npz_data['feature_is_matched'], dtype=np.int64)
    matched_gt_is_unknown = np.asarray(npz_data['feature_matched_gt_is_unknown'], dtype=np.int64)
    feature_specs = [
        ('objectness_features', 'Objectness features'),
        ('knownness_features', 'Knownness features'),
        ('classification_features', 'Classification features'),
    ]
    for dim in dims:
        for method in methods:
            for view_name, base_mask, base_labels, names, colors in build_view_specs(groups, is_matched, matched_gt_is_unknown):
                figure, axes = create_axes(dim, 3, (18, 5.8) if dim == 3 else (16.5, 5.2))
                plotted_any = False
                legend_handles = None
                legend_labels = None
                for axis, (key, title) in zip(axes, feature_specs):
                    if key not in npz_data:
                        axis.set_axis_off()
                        continue
                    features = np.asarray(npz_data[key], dtype=np.float32)
                    mask = base_mask.copy()
                    labels = base_labels[mask]
                    features = features[mask]
                    if view_name == 'semantic_known_unknown' and np.unique(labels).size < 2:
                        axis.set_axis_off()
                        continue
                    if features.shape[0] < max(8, dim * 3):
                        axis.set_axis_off()
                        continue
                    features, labels = subsample(features, labels, max_points)
                    embedding = compute_embedding(features, method, dim)
                    if embedding is None:
                        axis.set_axis_off()
                        continue
                    scatter_embedding(axis, embedding, labels, names, colors, dim)
                    axis.set_title(f'{title} · {method.upper()} · {dim}D · {view_name}')
                    handles, labels_text = axis.get_legend_handles_labels()
                    if handles:
                        legend_handles, legend_labels = handles, labels_text
                    plotted_any = True
                if plotted_any:
                    if legend_handles:
                        figure.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=max(2, len(legend_labels)), frameon=False)
                    figure.tight_layout()
                    figure.savefig(output_dir / f'feature_embedding_{method}_{dim}d_{view_name}.png', bbox_inches='tight')
                    plt.close(figure)
                else:
                    plt.close(figure)


def plot_score_space(stats, output_dir, methods, dims, max_points):
    features = np.stack([
        stats['objectness_probability'],
        stats['unknown_probability'],
        stats['max_known_class_probability'],
    ], axis=1).astype(np.float32)
    groups = stats['query_group'].astype(np.int64)
    is_matched = stats['is_matched'].astype(np.int64)
    matched_gt_is_unknown = stats['matched_gt_is_unknown'].astype(np.int64)
    for dim in dims:
        for method in methods:
            for view_name, base_mask, base_labels, names, colors in build_view_specs(groups, is_matched, matched_gt_is_unknown):
                masked_features = features[base_mask]
                labels = base_labels[base_mask]
                if view_name == 'semantic_known_unknown' and np.unique(labels).size < 2:
                    continue
                if masked_features.shape[0] < max(8, dim * 3):
                    continue
                masked_features, labels = subsample(masked_features, labels, max_points)
                embedding = compute_embedding(masked_features, method, dim)
                if embedding is None:
                    continue
                figure, axes = create_axes(dim, 1, (7.8, 6.2) if dim == 3 else (7.0, 5.8))
                axis = axes[0]
                scatter_embedding(axis, embedding, labels, names, colors, dim)
                axis.set_title(f'Score-space · {method.upper()} · {dim}D · {view_name}')
                axis.legend(frameon=False, fontsize=8)
                figure.tight_layout()
                figure.savefig(output_dir / f'score_space_{method}_{dim}d_{view_name}.png', bbox_inches='tight')
                plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser('Standalone PCA / t-SNE / UMAP plotting for UOD stats and features')
    parser.add_argument('--stats_dir', required=True, type=str, help='path like eval/visualizations/epoch_xxxx/stats')
    parser.add_argument('--max_points', default=1200, type=int)
    return parser


def main(args):
    stats_dir = Path(args.stats_dir)
    output_dir = stats_dir / 'standalone_manifold'
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = stats_dir / 'feature_samples.npz'
    csv_path = stats_dir / 'query_statistics.csv'
    methods = ['pca', 'tsne', 'umap']
    dims = [2, 3]

    if npz_path.exists():
        npz_data = load_feature_samples(npz_path)
        plot_feature_embeddings(npz_data, output_dir, methods, dims, args.max_points)
    if csv_path.exists():
        stats = load_query_statistics(csv_path)
        plot_score_space(stats, output_dir, methods, dims, args.max_points)

    print(f'Saved manifold plots to: {output_dir}')


if __name__ == '__main__':
    main(build_parser().parse_args())
