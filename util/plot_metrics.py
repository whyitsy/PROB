import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'cyan': '#33BBEE',
    'red': '#CC3311',
    'green': '#009988',
    'magenta': '#EE3377',
    'yellow': '#EEDD44',
    'purple': '#7A52A5',
    'gray': '#6C757D',
    'black': '#222222',
}


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def append_json_record(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


class _SeriesReader:
    def __init__(self, jsonl_path: Path):
        self.jsonl_path = Path(jsonl_path)

    def rows(self):
        if not self.jsonl_path.exists():
            return []
        rows = []
        for line in self.jsonl_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows


def _ema(values, alpha=0.15):
    out = []
    last = None
    for v in values:
        if v is None:
            out.append(last)
            continue
        if last is None:
            last = v
        else:
            last = alpha * v + (1.0 - alpha) * last
        out.append(last)
    return out


class MetricPlotter:
    def __init__(self, output_dir: Path, metrics_filename='metrics_log.jsonl', step_metrics_filename='metrics_step.jsonl'):
        self.output_dir = Path(output_dir)
        self.metrics_path = self.output_dir / metrics_filename
        self.step_metrics_path = self.output_dir / step_metrics_filename
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_rows = _SeriesReader(self.metrics_path).rows()
        self.step_rows = _SeriesReader(self.step_metrics_path).rows()

    def refresh_all(self):
        self.plot_epoch_metrics()
        self.plot_epoch_losses()
        self.plot_epoch_pseudo_stats()
        self.plot_epoch_decoupling()
        self.plot_step_losses()
        self.plot_step_stats()
        self.plot_step_aux_losses()

    def _save(self, fig, name):
        fig.savefig(self.plots_dir / name, dpi=220, bbox_inches='tight')
        plt.close(fig)

    def plot_epoch_metrics(self):
        if not self.epoch_rows:
            return
        history = []
        for row in self.epoch_rows:
            metrics = row.get('test_metrics') or {}
            epoch = row.get('epoch')
            if epoch is None:
                continue
            history.append({
                'epoch': int(epoch),
                'Current AP50': _safe_float(metrics.get('CK_AP50')),
                'Known AP50': _safe_float(metrics.get('K_AP50')),
                'Unknown Recall50': _safe_float(metrics.get('U_R50')),
                'WI@0.8': _safe_float(metrics.get('WI')),
                'A-OSE': _safe_float(metrics.get('AOSA', metrics.get('A-OSE'))),
            })
        if not history:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, color in [
            ('Current AP50', PALETTE['blue']),
            ('Known AP50', PALETTE['green']),
            ('Unknown Recall50', PALETTE['magenta']),
        ]:
            xs = [r['epoch'] for r in history if r.get(key) is not None]
            ys = [r[key] for r in history if r.get(key) is not None]
            if xs:
                ax.plot(xs, ys, marker='o', linewidth=2.2, color=color, label=key)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Open-World Percentage Metrics')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        self._save(fig, 'open_world_metric_trends_percent.png')

        fig, ax1 = plt.subplots(figsize=(10, 6))
        lines, labels = [], []
        xs = [r['epoch'] for r in history if r.get('WI@0.8') is not None]
        ys = [r['WI@0.8'] for r in history if r.get('WI@0.8') is not None]
        if xs:
            line = ax1.plot(xs, ys, marker='o', linewidth=2.2, color=PALETTE['orange'], label='WI@0.8')[0]
            lines.append(line)
            labels.append(line.get_label())
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Wilderness Impact')
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        xs = [r['epoch'] for r in history if r.get('A-OSE') is not None]
        ys = [r['A-OSE'] for r in history if r.get('A-OSE') is not None]
        if xs:
            line = ax2.plot(xs, ys, marker='s', linewidth=2.2, color=PALETTE['red'], label='A-OSE')[0]
            lines.append(line)
            labels.append(line.get_label())
        ax2.set_ylabel('Absolute Open-Set Error')
        if lines:
            ax1.legend(lines, labels, frameon=False, loc='best')
        ax1.set_title('Open-World Error Metrics')
        self._save(fig, 'open_world_metric_trends_openworld.png')

    def plot_epoch_losses(self):
        if not self.epoch_rows:
            return
        epochs = [int(r['epoch']) for r in self.epoch_rows if r.get('epoch') is not None]
        if not epochs:
            return
        total_loss = [_safe_float(r.get('train_loss')) for r in self.epoch_rows if r.get('epoch') is not None]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, total_loss, marker='o', linewidth=2.2, color=PALETTE['blue'], label='total_loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Total Loss Trend')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        self._save(fig, 'training_loss_total_trend.png')

        basic_series = [
            ('loss_ce', 'train_loss_ce_unscaled', PALETTE['blue']),
            ('loss_bbox', 'train_loss_bbox_unscaled', PALETTE['orange']),
            ('loss_giou', 'train_loss_giou_unscaled', PALETTE['green']),
            ('loss_obj_ll_scaled', 'train_loss_obj_ll', PALETTE['cyan']),
        ]
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, key, color in basic_series:
            ys = [_safe_float(r.get(key)) for r in self.epoch_rows if r.get('epoch') is not None]
            xs = [e for e, y in zip(epochs, ys) if y is not None]
            ys = [y for y in ys if y is not None]
            if xs:
                ax.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Base Detection Loss Components')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=2)
        self._save(fig, 'training_loss_base_components.png')

        ow_series = [
            ('unk_known', 'train_loss_unk_known_unscaled', PALETTE['orange']),
            ('obj_pseudo', 'train_loss_obj_pseudo_unscaled', PALETTE['blue']),
            ('unk_pseudo', 'train_loss_unk_pseudo_unscaled', PALETTE['magenta']),
            ('decorr', 'train_loss_decorr_unscaled', PALETTE['green']),
        ]
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, key, color in ow_series:
            ys = [_safe_float(r.get(key)) for r in self.epoch_rows if r.get('epoch') is not None]
            xs = [e for e, y in zip(epochs, ys) if y is not None]
            ys = [y for y in ys if y is not None]
            if xs:
                ax.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Open-World Core Loss Components')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        self._save(fig, 'training_loss_openworld_components.png')

    def plot_epoch_pseudo_stats(self):
        if not self.epoch_rows:
            return
        epochs = [int(r['epoch']) for r in self.epoch_rows if r.get('epoch') is not None]
        count_keys = [
            ('dummy_pos', 'train_stat_num_dummy_pos', PALETTE['blue']),
            ('valid_unmatched', 'train_stat_num_valid_unmatched', PALETTE['orange']),
            ('pos_candidates', 'train_stat_num_pos_candidates', PALETTE['green']),
            ('batch_selected_pos', 'train_stat_num_batch_selected_pos', PALETTE['magenta']),
        ]
        fig, ax = plt.subplots(figsize=(11, 6))
        plotted = False
        for label, key, color in count_keys:
            ys = [_safe_float(r.get(key)) for r in self.epoch_rows if r.get('epoch') is not None]
            xs = [e for e, y in zip(epochs, ys) if y is not None]
            ys = [y for y in ys if y is not None]
            if xs:
                plotted = True
                ax.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
        if plotted:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Count')
            ax.set_title('Pseudo-Supervision Count Statistics')
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
            self._save(fig, 'pseudo_stat_counts.png')
        else:
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        for label, key, color in [
            ('selection_ratio', 'train_pseudo_selection_ratio', PALETTE['cyan']),
            ('accept_ratio', 'train_pseudo_accept_ratio', PALETTE['red']),
        ]:
            ys = [_safe_float(r.get(key)) for r in self.epoch_rows if r.get('epoch') is not None]
            xs = [e for e, y in zip(epochs, ys) if y is not None]
            ys = [y for y in ys if y is not None]
            if xs:
                ax.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ratio')
        ax.set_title('Pseudo-Supervision Efficiency Trends')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        self._save(fig, 'pseudo_efficiency_trends.png')

    def plot_epoch_decoupling(self):
        if not self.epoch_rows:
            return
        keys = [
            ('corr_fg_obj_unk', PALETTE['blue']),
            ('corr_fg_obj_cls', PALETTE['orange']),
            ('corr_fg_unk_cls', PALETTE['green']),
            ('corr_global_obj_unk', PALETTE['magenta']),
            ('corr_global_obj_cls', PALETTE['cyan']),
            ('corr_global_unk_cls', PALETTE['red']),
        ]
        fig, ax = plt.subplots(figsize=(11, 6.5))
        plotted = False
        for key, color in keys:
            xs, ys = [], []
            for row in self.epoch_rows:
                epoch = row.get('epoch')
                metrics = row.get('test_metrics') or {}
                val = _safe_float(metrics.get(key))
                if epoch is None or val is None:
                    continue
                xs.append(int(epoch))
                ys.append(val)
            if xs:
                plotted = True
                ax.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=key)
        if plotted:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Pearson Correlation')
            ax.set_title('Decoupling Correlation Trends')
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, fontsize=9, ncol=2)
            self._save(fig, 'decoupling_correlation_trends.png')
        else:
            plt.close(fig)

    def plot_step_losses(self):
        if not self.step_rows:
            return
        steps = [int(r['global_step']) for r in self.step_rows if r.get('global_step') is not None]
        if not steps:
            return
        self._plot_step_group(
            'training_loss_total_step.png',
            'Step-level Total Loss',
            steps,
            [('total_loss', PALETTE['blue'])],
            ['train/total_loss'],
            ylabel='Loss',
        )
        self._plot_step_group(
            'training_loss_base_step.png',
            'Step-level Base Loss Components',
            steps,
            [
                ('loss_ce', PALETTE['blue']),
                ('loss_bbox', PALETTE['orange']),
                ('loss_giou', PALETTE['green']),
                ('loss_obj_ll_scaled', PALETTE['cyan']),
            ],
            [
                'train_unscaled/loss_ce',
                'train_unscaled/loss_bbox',
                'train_unscaled/loss_giou',
                'train_scaled/loss_obj_ll',
            ],
            ylabel='Loss',
        )
        self._plot_step_group(
            'training_loss_openworld_step.png',
            'Step-level Open-World Loss Components',
            steps,
            [
                ('loss_unk_known', PALETTE['orange']),
                ('loss_obj_pseudo', PALETTE['blue']),
                ('loss_unk_pseudo', PALETTE['magenta']),
                ('loss_decorr', PALETTE['green']),
            ],
            [
                'train_unscaled/loss_unk_known',
                'train_unscaled/loss_obj_pseudo',
                'train_unscaled/loss_unk_pseudo',
                'train_unscaled/loss_decorr',
            ],
            ylabel='Loss',
        )
        self._plot_step_group(
            'training_objectness_stats_step.png',
            'Objectness / Unknownness Score Trends',
            steps,
            [
                ('obj_prob_matched_mean', PALETTE['blue']),
                ('obj_prob_unmatched_mean', PALETTE['orange']),
                ('unk_prob_mean', PALETTE['magenta']),
                ('cls_max_mean', PALETTE['green']),
            ],
            [
                'train_stats/obj_prob_matched_mean',
                'train_stats/obj_prob_unmatched_mean',
                'train_stats/unk_prob_mean',
                'train_stats/cls_max_mean',
            ],
            ylabel='Value',
        )
        self._plot_step_group(
            'training_score_calibration_step.png',
            'Learnable Score Calibration Coefficients',
            steps,
            [
                ('known_unk_suppress', PALETTE['cyan']),
                ('unknown_known_suppress', PALETTE['red']),
            ],
            [
                'train_stats/known_unk_suppress_coeff',
                'train_stats/unknown_known_suppress_coeff',
            ],
            ylabel='Coefficient',
        )

    def plot_step_stats(self):
        if not self.step_rows:
            return
        steps = [int(r['global_step']) for r in self.step_rows if r.get('global_step') is not None]
        self._plot_step_group(
            'training_pseudo_stats_step.png',
            'Step-level Pseudo Mining Statistics',
            steps,
            [
                ('dummy_pos', PALETTE['blue']),
                ('valid_unmatched', PALETTE['orange']),
                ('pos_candidates', PALETTE['green']),
                ('batch_selected_pos', PALETTE['magenta']),
                ('selection_ratio', PALETTE['cyan']),
                ('accept_ratio', PALETTE['red']),
            ],
            [
                'train_stats/stat_num_dummy_pos',
                'train_stats/stat_num_valid_unmatched',
                'train_stats/stat_num_pos_candidates',
                'train_stats/stat_num_batch_selected_pos',
                'train_stats/pseudo_selection_ratio',
                'train_stats/pseudo_accept_ratio',
            ],
            ylabel='Value',
        )
        self._plot_step_group(
            'training_gate_stats_step.png',
            'ODQE Gate / Soft-Attention Trends',
            steps,
            [
                ('gate_mean', PALETTE['purple']),
                ('cls_attn_mean', PALETTE['yellow']),
            ],
            [
                'train_stats/odqe_gate_mean',
                'train_stats/stat_cls_attn_mean',
            ],
            ylabel='Value',
        )

    def plot_step_aux_losses(self):
        if not self.step_rows:
            return
        keys = sorted({k for row in self.step_rows for k in row.keys() if k.startswith('train_unscaled/loss_obj_pseudo_') or k.startswith('train_unscaled/loss_unk_pseudo_') or k.startswith('train_unscaled/loss_decorr_')})
        if not keys:
            return
        fig, ax = plt.subplots(figsize=(12, 6.5))
        steps = [int(r['global_step']) for r in self.step_rows if r.get('global_step') is not None]
        colors = [PALETTE['blue'], PALETTE['orange'], PALETTE['green'], PALETTE['magenta'], PALETTE['cyan'], PALETTE['red'], PALETTE['purple']]
        plotted = False
        for idx, key in enumerate(keys):
            xs, ys = [], []
            for row in self.step_rows:
                step = row.get('global_step')
                val = _safe_float(row.get(key))
                if step is None or val is None:
                    continue
                xs.append(int(step))
                ys.append(val)
            if xs:
                plotted = True
                smooth = _ema(ys, alpha=0.08)
                ax.plot(xs, smooth, linewidth=1.8, color=colors[idx % len(colors)], label=key.replace('train_unscaled/', ''))
        if plotted:
            ax.set_xlabel('Global Step')
            ax.set_ylabel('Loss')
            ax.set_title('Hierarchical AUX Loss Trends')
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, fontsize=8, ncol=2)
            self._save(fig, 'training_aux_loss_trends_step.png')
        else:
            plt.close(fig)

    def _plot_step_group(self, filename, title, all_steps, named_series, keys, ylabel='Value'):
        fig, ax = plt.subplots(figsize=(12, 6.5))
        plotted = False
        for (label, color), key in zip(named_series, keys):
            xs, ys = [], []
            for row in self.step_rows:
                step = row.get('global_step')
                val = _safe_float(row.get(key))
                if step is None or val is None:
                    continue
                xs.append(int(step))
                ys.append(val)
            if xs:
                plotted = True
                ax.plot(xs, ys, alpha=0.18, linewidth=0.9, color=color)
                ax.plot(xs, _ema(ys, alpha=0.08), linewidth=2.0, color=color, label=label)
        if plotted:
            ax.set_xlabel('Global Step')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, ncol=2)
            self._save(fig, filename)
        else:
            plt.close(fig)


def refresh_metric_plots(output_dir: Path, metrics_filename='metrics_log.jsonl', step_metrics_filename='metrics_step.jsonl'):
    MetricPlotter(Path(output_dir), metrics_filename=metrics_filename, step_metrics_filename=step_metrics_filename).refresh_all()
