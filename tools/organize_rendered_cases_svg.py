import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        'Figure atlas generation is disabled by default because figure_atlas duplicates rendered_cases.'
    )
    parser.add_argument('--render_manifest', required=True, type=str, help='kept for CLI compatibility')
    parser.add_argument('--representative_manifest', default=None, type=str, help='kept for CLI compatibility')
    parser.add_argument('--output_dir', required=True, type=str, help='kept for CLI compatibility')
    parser.add_argument('--per_group_limit', default=3, type=int)
    parser.add_argument('--chapter3_categories', default='unknown', type=str)
    parser.add_argument('--chapter3_modes', default='sampling,trajectory', type=str)
    parser.add_argument('--chapter4_categories', default='known,unknown,odqe_salient', type=str)
    parser.add_argument('--chapter4_modes', default='gate,joint,trajectory', type=str)
    parser.add_argument(
        '--enable_atlas',
        action='store_true',
        help='optional escape hatch; still disabled in this cleaned workflow unless you restore the old organizer.',
    )
    return parser


def main(args):
    output_dir = Path(args.output_dir)
    if args.enable_atlas:
        raise RuntimeError(
            'figure_atlas generation has been intentionally disabled in this cleaned workflow. '
            'If you really need atlas boards again, restore the previous organizer implementation explicitly.'
        )
    print(
        'Skip figure_atlas generation: rendered_cases already contains the per-case mechanism figures, '
        'and atlas boards only duplicate those outputs.'
    )
    print(f'No files were written under: {output_dir}')


if __name__ == '__main__':
    main(build_parser().parse_args())
