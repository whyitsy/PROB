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
}


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def append_json_record(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False) + '\n')


class _JsonlSeriesReader:
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
    smoothed = []
    previous = None
    for value in values:
        if value is None:
            smoothed.append(previous)
            continue
        if previous is None:
            previous = value
        else:
            previous = alpha * value + (1.0 - alpha) * previous
        smoothed.append(previous)
    return smoothed


class ExperimentMetricsPlotter:
    def __init__(
        self,
        output_dir: Path,
        train_epoch_metrics_file='train/metrics_epoch.jsonl',
        eval_epoch_metrics_file='eval/metrics_epoch.jsonl',
        train_step_metrics_file='train/metrics_step.jsonl',
    ):
        self.output_dir = Path(output_dir)
        self.train_epoch_metrics_path = self.output_dir / train_epoch_metrics_file
        self.eval_epoch_metrics_path = self.output_dir / eval_epoch_metrics_file
        self.train_step_metrics_path = self.output_dir / train_step_metrics_file
        self.train_plots_dir = self.output_dir / 'train' / 'plots'
        self.eval_plots_dir = self.output_dir / 'eval' / 'plots'
        self.train_plots_dir.mkdir(parents=True, exist_ok=True)
        self.eval_plots_dir.mkdir(parents=True, exist_ok=True)
        self.train_epoch_rows = _JsonlSeriesReader(self.train_epoch_metrics_path).rows()
        self.eval_epoch_rows = _JsonlSeriesReader(self.eval_epoch_metrics_path).rows()
        self.train_step_rows = _JsonlSeriesReader(self.train_step_metrics_path).rows()

    def refresh_all(self):
        self.plot_eval_open_world_metrics()
        self.plot_epoch_training_losses()
        self.plot_epoch_pseudo_statistics()
        self.plot_epoch_branch_correlation_metrics()
        self.plot_step_training_losses()
        self.plot_step_pseudo_statistics()
        self.plot_step_auxiliary_losses()

    def _save(self, figure, file_stem, split='train'):
        output_dir = self.train_plots_dir if split == 'train' else self.eval_plots_dir
        figure.savefig(output_dir / f'{file_stem}.svg', bbox_inches='tight')
        plt.close(figure)

    def plot_eval_open_world_metrics(self):
        if not self.eval_epoch_rows:
            return
        history = []
        for row in self.eval_epoch_rows:
            epoch = row.get('epoch')
            metrics = row.get('open_world_metrics') or row.get('test_metrics') or {}
            if epoch is None:
                continue
            history.append({
                'epoch': int(epoch),
                'current_ap50': _safe_float(metrics.get('CK_AP50')),
                'known_ap50': _safe_float(metrics.get('K_AP50')),
                'unknown_recall50': _safe_float(metrics.get('U_R50')),
                'wilderness_impact': _safe_float(metrics.get('WI')),
                'absolute_open_set_error': _safe_float(metrics.get('AOSA', metrics.get('A-OSE'))),
            })
        if not history:
            return

        figure, axis = plt.subplots(figsize=(10, 6))
        for label, key, color in [
            ('Current AP50', 'current_ap50', PALETTE['blue']),
            ('Known AP50', 'known_ap50', PALETTE['green']),
            ('Unknown Recall50', 'unknown_recall50', PALETTE['magenta']),
        ]:
            xs = [item['epoch'] for item in history if item[key] is not None]
            ys = [item[key] for item in history if item[key] is not None]
            if xs:
                axis.plot(xs, ys, marker='o', linewidth=2.2, color=color, label=label)
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Percentage (%)')
        axis.set_title('Open-World Detection Percentage Metrics')
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        self._save(figure, 'open_world_percentage_metrics', split='eval')

        figure, left_axis = plt.subplots(figsize=(10, 6))
        lines = []
        labels = []
        xs = [item['epoch'] for item in history if item['wilderness_impact'] is not None]
        ys = [item['wilderness_impact'] for item in history if item['wilderness_impact'] is not None]
        if xs:
            line = left_axis.plot(xs, ys, marker='o', linewidth=2.2, color=PALETTE['orange'], label='WI@0.8')[0]
            lines.append(line)
            labels.append(line.get_label())
        right_axis = left_axis.twinx()
        xs = [item['epoch'] for item in history if item['absolute_open_set_error'] is not None]
        ys = [item['absolute_open_set_error'] for item in history if item['absolute_open_set_error'] is not None]
        if xs:
            line = right_axis.plot(xs, ys, marker='s', linewidth=2.2, color=PALETTE['red'], label='A-OSE')[0]
            lines.append(line)
            labels.append(line.get_label())
        left_axis.set_xlabel('Epoch')
        left_axis.set_ylabel('Wilderness Impact')
        right_axis.set_ylabel('Absolute Open-Set Error')
        left_axis.set_title('Open-World Error Metrics')
        left_axis.grid(True, alpha=0.25)
        if lines:
            left_axis.legend(lines, labels, frameon=False)
        self._save(figure, 'open_world_error_metrics', split='eval')

    def plot_epoch_training_losses(self):
        if not self.train_epoch_rows:
            return
        epochs = [int(row['epoch']) for row in self.train_epoch_rows if row.get('epoch') is not None]
        if not epochs:
            return
        total_losses = [_safe_float(row.get('train_total_loss')) for row in self.train_epoch_rows if row.get('epoch') is not None]
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(epochs, total_losses, marker='o', linewidth=2.2, color=PALETTE['blue'], label='total_loss')
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Loss')
        axis.set_title('Training Total Loss Trend')
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        self._save(figure, 'training_total_loss', split='train')

        base_series = [
            ('classification', 'train_raw_loss_ce', PALETTE['blue']),
            ('box_l1', 'train_raw_loss_bbox', PALETTE['orange']),
            ('giou', 'train_raw_loss_giou', PALETTE['green']),
            ('matched_objectness', 'train_weighted_loss_obj_ll', PALETTE['cyan']),
        ]
        figure, axis = plt.subplots(figsize=(11, 6))
        for label, key, color in base_series:
            ys = [_safe_float(row.get(key)) for row in self.train_epoch_rows if row.get('epoch') is not None]
            xs = [epoch for epoch, value in zip(epochs, ys) if value is not None]
            ys = [value for value in ys if value is not None]
            if xs:
                axis.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Loss')
        axis.set_title('Base Detection Loss Components')
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, ncol=2)
        self._save(figure, 'training_base_loss_components', split='train')

        uod_series = [
            ('matched_known_knownness', 'train_raw_loss_unk_known', PALETTE['orange']),
            ('pseudo_positive_objectness', 'train_raw_loss_obj_pseudo', PALETTE['blue']),
            ('pseudo_unknown_knownness', 'train_raw_loss_unk_pseudo', PALETTE['magenta']),
            ('branch_decorrelation', 'train_raw_loss_decorr', PALETTE['green']),
        ]
        figure, axis = plt.subplots(figsize=(11, 6))
        plotted = False
        for label, key, color in uod_series:
            ys = [_safe_float(row.get(key)) for row in self.train_epoch_rows if row.get('epoch') is not None]
            xs = [epoch for epoch, value in zip(epochs, ys) if value is not None]
            ys = [value for value in ys if value is not None]
            if xs:
                plotted = True
                axis.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
        if plotted:
            axis.set_xlabel('Epoch')
            axis.set_ylabel('Loss')
            axis.set_title('Open-World Loss Components')
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False)
            self._save(figure, 'training_open_world_loss_components', split='train')
        else:
            plt.close(figure)

    def plot_epoch_pseudo_statistics(self):
        if not self.train_epoch_rows:
            return
        epochs = [int(row['epoch']) for row in self.train_epoch_rows if row.get('epoch') is not None]
        count_keys = [
            ('selected_pseudo_positive_queries', 'num_selected_pseudo_positive_queries', PALETTE['blue']),
            ('reliable_background_queries', 'num_selected_reliable_background_queries', PALETTE['orange']),
            ('candidate_queries', 'num_pseudo_positive_candidates', PALETTE['green']),
            ('ignored_queries', 'num_classification_ignored_queries', PALETTE['magenta']),
        ]
        figure, axis = plt.subplots(figsize=(11, 6))
        plotted = False
        for label, key, color in count_keys:
            ys = [_safe_float(row.get(key)) for row in self.train_epoch_rows if row.get('epoch') is not None]
            xs = [epoch for epoch, value in zip(epochs, ys) if value is not None]
            ys = [value for value in ys if value is not None]
            if xs:
                plotted = True
                axis.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
        if plotted:
            axis.set_xlabel('Epoch')
            axis.set_ylabel('Count')
            axis.set_title('Pseudo Mining Count Statistics')
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False)
            self._save(figure, 'pseudo_mining_counts', split='train')
        else:
            plt.close(figure)

        figure, axis = plt.subplots(figsize=(10, 6))
        for label, key, color in [
            ('selection_ratio', 'pseudo_positive_selection_ratio', PALETTE['cyan']),
            ('accept_ratio', 'pseudo_positive_accept_ratio', PALETTE['red']),
        ]:
            ys = [_safe_float(row.get(key)) for row in self.train_epoch_rows if row.get('epoch') is not None]
            xs = [epoch for epoch, value in zip(epochs, ys) if value is not None]
            ys = [value for value in ys if value is not None]
            if xs:
                axis.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Ratio')
        axis.set_title('Pseudo Mining Efficiency')
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        self._save(figure, 'pseudo_mining_efficiency', split='train')

    def plot_epoch_branch_correlation_metrics(self):
        if not self.eval_epoch_rows:
            return
        figure, axis = plt.subplots(figsize=(11, 6.5))
        plotted = False
        for key, color in [
            ('corr_fg_obj_unk', PALETTE['blue']),
            ('corr_fg_obj_cls', PALETTE['orange']),
            ('corr_fg_unk_cls', PALETTE['green']),
            ('corr_global_obj_unk', PALETTE['magenta']),
            ('corr_global_obj_cls', PALETTE['cyan']),
            ('corr_global_unk_cls', PALETTE['red']),
        ]:
            xs = []
            ys = []
            for row in self.eval_epoch_rows:
                epoch = row.get('epoch')
                metrics = row.get('open_world_metrics') or row.get('test_metrics') or {}
                value = _safe_float(metrics.get(key))
                if epoch is None or value is None:
                    continue
                xs.append(int(epoch))
                ys.append(value)
            if xs:
                plotted = True
                axis.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=key)
        if plotted:
            axis.set_xlabel('Epoch')
            axis.set_ylabel('Pearson Correlation')
            axis.set_title('Branch Correlation Trends')
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize=9, ncol=2)
            self._save(figure, 'branch_correlation_trends', split='eval')
        else:
            plt.close(figure)

    def plot_step_training_losses(self):
        if not self.train_step_rows:
            return
        self._plot_step_group(
            file_stem='step_total_loss',
            title='Step-level Total Loss',
            series=[('total_loss', 'train/loss/total', PALETTE['blue'])],
            ylabel='Loss',
        )
        self._plot_step_group(
            file_stem='step_base_losses',
            title='Step-level Base Detection Losses',
            series=[
                ('classification', 'train/loss_raw/loss_ce', PALETTE['blue']),
                ('box_l1', 'train/loss_raw/loss_bbox', PALETTE['orange']),
                ('giou', 'train/loss_raw/loss_giou', PALETTE['green']),
                ('matched_objectness', 'train/loss_weighted/loss_obj_ll', PALETTE['cyan']),
            ],
            ylabel='Loss',
        )
        self._plot_step_group(
            file_stem='step_open_world_losses',
            title='Step-level Open-World Losses',
            series=[
                ('matched_known_knownness', 'train/loss_raw/loss_unk_known', PALETTE['orange']),
                ('pseudo_positive_objectness', 'train/loss_raw/loss_obj_pseudo', PALETTE['blue']),
                ('pseudo_unknown_knownness', 'train/loss_raw/loss_unk_pseudo', PALETTE['magenta']),
                ('branch_decorrelation', 'train/loss_raw/loss_decorr', PALETTE['green']),
            ],
            ylabel='Loss',
        )
        self._plot_step_group(
            file_stem='step_query_score_statistics',
            title='Step-level Query Score Statistics',
            series=[
                ('matched_objectness_prob', 'train/query_stats/matched_objectness_prob_mean', PALETTE['blue']),
                ('unmatched_objectness_prob', 'train/query_stats/unmatched_objectness_prob_mean', PALETTE['orange']),
                ('unknown_probability', 'train/query_stats/unknown_probability_mean', PALETTE['magenta']),
                ('max_known_class_probability', 'train/query_stats/max_known_class_probability_mean', PALETTE['green']),
            ],
            ylabel='Value',
        )

    def plot_step_pseudo_statistics(self):
        if not self.train_step_rows:
            return
        self._plot_step_group(
            file_stem='step_pseudo_mining_statistics',
            title='Step-level Pseudo Mining Statistics',
            series=[
                ('selected_pseudo_positive_queries', 'train/loss_raw/num_selected_pseudo_positive_queries', PALETTE['blue']),
                ('candidate_queries', 'train/loss_raw/num_pseudo_positive_candidates', PALETTE['green']),
                ('selection_ratio', 'train/pseudo/selection_ratio', PALETTE['cyan']),
                ('accept_ratio', 'train/pseudo/accept_ratio', PALETTE['red']),
            ],
            ylabel='Value',
        )

    def plot_step_auxiliary_losses(self):
        if not self.train_step_rows:
            return
        aux_keys = sorted({
            key for row in self.train_step_rows for key in row.keys()
            if key.startswith('train/loss_raw/loss_obj_pseudo_')
            or key.startswith('train/loss_raw/loss_unk_pseudo_')
            or key.startswith('train/loss_raw/loss_decorr_')
        })
        if not aux_keys:
            return
        figure, axis = plt.subplots(figsize=(12, 6.5))
        colors = [PALETTE['blue'], PALETTE['orange'], PALETTE['green'], PALETTE['magenta'], PALETTE['cyan'], PALETTE['red'], PALETTE['purple']]
        plotted = False
        for index, key in enumerate(aux_keys):
            xs = []
            ys = []
            for row in self.train_step_rows:
                step = row.get('global_step')
                value = _safe_float(row.get(key))
                if step is None or value is None:
                    continue
                xs.append(int(step))
                ys.append(value)
            if xs:
                plotted = True
                axis.plot(xs, _ema(ys, alpha=0.08), linewidth=1.8, color=colors[index % len(colors)], label=key.replace('train/loss_raw/', ''))
        if plotted:
            axis.set_xlabel('Global Step')
            axis.set_ylabel('Loss')
            axis.set_title('Step-level Auxiliary Loss Trends')
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize=8, ncol=2)
            self._save(figure, 'step_auxiliary_loss_trends', split='train')
        else:
            plt.close(figure)

    def _plot_step_group(self, file_stem, title, series, ylabel='Value'):
        figure, axis = plt.subplots(figsize=(12, 6.5))
        plotted = False
        for label, key, color in series:
            xs = []
            ys = []
            for row in self.train_step_rows:
                step = row.get('global_step')
                value = _safe_float(row.get(key))
                if step is None or value is None:
                    continue
                xs.append(int(step))
                ys.append(value)
            if xs:
                plotted = True
                axis.plot(xs, ys, alpha=0.18, linewidth=0.9, color=color)
                axis.plot(xs, _ema(ys, alpha=0.08), linewidth=2.0, color=color, label=label)
        if plotted:
            axis.set_xlabel('Global Step')
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, ncol=2)
            self._save(figure, file_stem, split='train')
        else:
            plt.close(figure)


def refresh_metric_plots(
    output_dir: Path,
    train_epoch_metrics_file='train/metrics_epoch.jsonl',
    eval_epoch_metrics_file='eval/metrics_epoch.jsonl',
    train_step_metrics_file='train/metrics_step.jsonl',
):
    ExperimentMetricsPlotter(
        Path(output_dir),
        train_epoch_metrics_file=train_epoch_metrics_file,
        eval_epoch_metrics_file=eval_epoch_metrics_file,
        train_step_metrics_file=train_step_metrics_file,
    ).refresh_all()
