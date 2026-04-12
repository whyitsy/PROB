import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.figure_svg_utils import save_svg_figure

GROUP_NAMES = ['matched-known', 'unmatched-high-unknown', 'other-unmatched']
GROUP_COLORS = ['#00A65A', '#D81B60', '#6C757D']


def load_query_statistics(csv_path):
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
    if features.shape[0] <= max_points:
        return features, groups
    indices = np.linspace(0, features.shape[0] - 1, max_points).astype(np.int64)
    return features[indices], groups[indices]


def plot_feature_manifold_3d(npz_path, output_dir, max_points):
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
        axis = fig.add_subplot(1, 3, subplot_index, projection='3d')
        for group_index, (group_name, color) in enumerate(zip(GROUP_NAMES, GROUP_COLORS)):
            mask = feature_groups == group_index
            if np.any(mask):
                axis.scatter(projection[mask, 0], projection[mask, 1], projection[mask, 2], s=10, alpha=0.62, c=color, label=group_name)
        axis.set_title(title)
        axis.set_xlabel('PC1')
        axis.set_ylabel('PC2')
        axis.set_zlabel('PC3')
    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    save_svg_figure(fig, output_dir / 'feature_manifold_3d.svg')


def plot_score_space_3d(stats, output_dir, max_points):
    features = np.stack([
        stats['objectness_probability'],
        stats['unknown_probability'],
        stats['max_known_class_probability'],
    ], axis=1)
    features, groups = subsample(features, stats['query_group'], max_points)
    fig = plt.figure(figsize=(8.4, 6.8), constrained_layout=True)
    axis = fig.add_subplot(1, 1, 1, projection='3d')
    for group_index, (group_name, color) in enumerate(zip(GROUP_NAMES, GROUP_COLORS)):
        mask = groups == group_index
        if np.any(mask):
            axis.scatter(features[mask, 0], features[mask, 1], features[mask, 2], s=10, alpha=0.62, c=color, label=group_name)
    axis.set_xlabel('objectness prob')
    axis.set_ylabel('unknown prob')
    axis.set_zlabel('max known prob')
    axis.set_title('Score-space decision geometry (3D)')
    axis.legend(frameon=False, fontsize=9, loc='upper left')
    save_svg_figure(fig, output_dir / 'score_space_3d.svg')


def plot_score_slices(stats, output_dir, max_points):
    features = np.stack([
        stats['objectness_probability'],
        stats['unknown_probability'],
        stats['max_known_class_probability'],
    ], axis=1)
    features, groups = subsample(features, stats['query_group'], max_points)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
    slice_specs = [
        ((0, 1), 'objectness prob', 'unknown prob', 'Objectness vs Unknownness'),
        ((0, 2), 'objectness prob', 'max known prob', 'Objectness vs Max Known'),
        ((1, 2), 'unknown prob', 'max known prob', 'Unknownness vs Max Known'),
    ]
    for axis, ((x_idx, y_idx), x_label, y_label, title) in zip(axes, slice_specs):
        for group_index, (group_name, color) in enumerate(zip(GROUP_NAMES, GROUP_COLORS)):
            mask = groups == group_index
            if np.any(mask):
                axis.scatter(features[mask, x_idx], features[mask, y_idx], s=10, alpha=0.62, c=color, label=group_name)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    save_svg_figure(fig, output_dir / 'score_space_slices.svg')


def build_parser():
    parser = argparse.ArgumentParser('SVG 3D manifold and decision boundary plotting')
    parser.add_argument('--stats_dir', required=True, type=str, help='path like eval/visualizations/epoch_xxxx/stats')
    parser.add_argument('--max_points', default=1500, type=int)
    return parser


def main(args):
    stats_dir = Path(args.stats_dir)
    output_dir = stats_dir / 'standalone_3d'
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = stats_dir / 'feature_samples.npz'
    csv_path = stats_dir / 'query_statistics.csv'

    if npz_path.exists():
        plot_feature_manifold_3d(npz_path, output_dir, args.max_points)
    if csv_path.exists():
        stats = load_query_statistics(csv_path)
        plot_score_space_3d(stats, output_dir, args.max_points)
        plot_score_slices(stats, output_dir, args.max_points)

    print(f'Saved SVG 3D plots to: {output_dir}')


if __name__ == '__main__':
    main(build_parser().parse_args())
