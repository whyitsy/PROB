import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _safe_kde(ax, df, x, hue=None, title=''):
    if df.empty:
        ax.set_title(title + ' (no data)')
        return
    sns.kdeplot(data=df, x=x, hue=hue, fill=True, common_norm=False, alpha=0.35, ax=ax)
    ax.set_title(title)
    ax.set_xlim(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser('Plot score distributions from score_records.jsonl')
    parser.add_argument('--records', required=True, help='Path to score_records.jsonl')
    parser.add_argument('--output_dir', default='score_plots', help='Directory to save figures')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(Path(args.records))
    if df.empty:
        raise RuntimeError('No records loaded')

    overall_path = out_dir / 'score_density_known_vs_unknown.png'
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    _safe_kde(ax, df, x='score', hue='pred_type', title='Score density: predicted known vs predicted unknown')
    fig.tight_layout()
    fig.savefig(overall_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    matched_df = df[df['matched'] == True].copy()
    matched_path = out_dir / 'score_density_matched_gt_type.png'
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    _safe_kde(ax, matched_df, x='score', hue='matched_gt_type', title='Score density: matched GT known vs GT unknown')
    fig.tight_layout()
    fig.savefig(matched_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    ao_df = df[(df['pred_type'] == 'known') & (df['matched_gt_type'] == 'unknown')].copy()
    ao_path = out_dir / 'score_density_aose_like.png'
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    _safe_kde(ax, ao_df, x='score', title='A-OSE-like subset: GT unknown but predicted known')
    fig.tight_layout()
    fig.savefig(ao_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    bar_path = out_dir / 'count_matrix_like.png'
    pivot = pd.crosstab(df['matched_gt_type'], df['pred_type'])
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Detection count matrix (GT type vs predicted type)')
    fig.tight_layout()
    fig.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('[OK] saved plots to', out_dir)


if __name__ == '__main__':
    main()
