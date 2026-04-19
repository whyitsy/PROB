import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.visual.embeddings import render_saved_eval_embeddings
from visual.viz_config import build_viz_cfg


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(',') if item.strip()]


def parse_str_list(text):
    return [item.strip() for item in text.split(',') if item.strip()]


def build_parser():
    parser = argparse.ArgumentParser('Offline renderer for eval embedding plots')
    parser.add_argument('--stats_dir', required=True, type=str, help='path like output/eval/visualizations/epoch_0004/stats')
    parser.add_argument('--methods', default='pca,tsne,umap', type=str, help='comma separated embedding methods')
    parser.add_argument('--dims', default='2,3', type=str, help='comma separated embedding dims')
    parser.add_argument('--figure_format', default='svg', type=str)
    return parser


def main(args):
    viz_cfg = build_viz_cfg(True)
    viz_cfg['save_feature_embedding_plots'] = True
    viz_cfg['embedding_methods'] = parse_str_list(args.methods)
    viz_cfg['embedding_dims'] = parse_int_list(args.dims)
    viz_cfg['figure_format'] = args.figure_format

    output_dirs = render_saved_eval_embeddings(args.stats_dir, viz_cfg)
    print(f'Rendered offline eval embeddings from: {args.stats_dir}')
    for key, value in output_dirs.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main(build_parser().parse_args())
