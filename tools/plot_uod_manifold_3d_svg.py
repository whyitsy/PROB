import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.visual.embeddings import (
    load_query_statistics,
    plot_feature_manifold_3d,
    plot_score_space_3d,
    plot_score_space_slices,
)


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
        plot_feature_manifold_3d(npz_path, output_dir / 'feature_manifold_3d.svg', args.max_points)
    if csv_path.exists():
        stats = load_query_statistics(csv_path)
        plot_score_space_3d(stats, output_dir / 'score_space_3d.svg', args.max_points)
        plot_score_space_slices(stats, output_dir / 'score_space_slices.svg', args.max_points)

    print(f'Saved SVG 3D plots to: {output_dir}')


if __name__ == '__main__':
    main(build_parser().parse_args())
