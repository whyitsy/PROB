import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def reshape_for_heatmap(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v).reshape(-1)
    cols = max(1, int(math.sqrt(len(v))))
    rows = math.ceil(len(v) / cols)
    padded = np.pad(v, (0, rows * cols - len(v)), constant_values=np.nan)
    return padded.reshape(rows, cols)


def plot_array(arr: np.ndarray, title: str, path: Path, cmap: str = 'YlOrRd') -> None:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = reshape_for_heatmap(arr)
    elif arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = reshape_for_heatmap(arr)
    plt.figure(figsize=(6, 5))
    sns.heatmap(arr, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser('Plot attention / debug heatmaps from .npz files')
    parser.add_argument('--npz', required=True, help='Path to .npz file produced by inference_visualize.py or collect_eval_debug.py')
    parser.add_argument('--output_dir', default='attention_plots', help='Directory to save heatmaps')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz, allow_pickle=True)
    for key in data.files:
        arr = data[key]
        out_path = out_dir / f'{Path(args.npz).stem}_{key}.png'
        plot_array(arr, key, out_path)
        print(f'[OK] {out_path}')


if __name__ == '__main__':
    main()
