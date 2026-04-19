
import json
from pathlib import Path

from util.visual.evaluation import (
    plot_branch_correlation_trends,
    plot_open_world_error_metrics,
    plot_open_world_percentage_metrics,
)
from util.visual.training import (
    plot_pseudo_mining_counts,
    plot_pseudo_mining_efficiency,
    plot_step_auxiliary_loss_trends,
    plot_step_base_losses,
    plot_step_open_world_losses,
    plot_step_pseudo_mining_counts_bars,
    plot_step_pseudo_mining_statistics,
    plot_step_query_score_statistics,
    plot_step_total_loss,
    plot_training_base_loss_components,
    plot_training_matched_objectness_loss_component,
    plot_training_open_world_loss_components,
    plot_training_total_loss,
)


def append_json_record(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False) + '\n')


class JsonlSeriesReader:
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


class ExperimentMetricsPlotter:
    step_bar_interval = 1000

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
        self.train_epoch_rows = JsonlSeriesReader(self.train_epoch_metrics_path).rows()
        self.eval_epoch_rows = JsonlSeriesReader(self.eval_epoch_metrics_path).rows()
        self.train_step_rows = JsonlSeriesReader(self.train_step_metrics_path).rows()

    def refresh_all(self):
        self.refresh_eval_plots()
        self.refresh_epoch_training_plots()
        self.refresh_epoch_pseudo_plots()
        self.refresh_step_training_plots()
        self.refresh_step_pseudo_plots()
        self.refresh_step_auxiliary_plots()

    def refresh_eval_plots(self):
        if not self.eval_epoch_rows:
            return
        plot_open_world_percentage_metrics(self.eval_epoch_rows, self.eval_plots_dir / 'open_world_percentage_metrics.svg')
        plot_open_world_error_metrics(self.eval_epoch_rows, self.eval_plots_dir / 'open_world_error_metrics.svg')
        plot_branch_correlation_trends(self.eval_epoch_rows, self.eval_plots_dir / 'branch_correlation_trends.svg')

    def refresh_epoch_training_plots(self):
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

        figure, axis = plt.subplots(figsize=(11, 6))
        for label, key, color in [
            ('classification', 'train_raw_loss_ce', PALETTE['blue']),
            ('box_l1', 'train_raw_loss_bbox', PALETTE['orange']),
            ('giou', 'train_raw_loss_giou', PALETTE['green']),
        ]:
            ys = [_safe_float(row.get(key)) for row in self.train_epoch_rows if row.get('epoch') is not None]
            xs = [epoch for epoch, value in zip(epochs, ys) if value is not None]
            ys = [value for value in ys if value is not None]
            if xs:
                axis.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Raw loss')
        axis.set_title('Base Detection Loss Components')
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, ncol=2)
        self._save(figure, 'training_base_loss_components', split='train')

        figure, axis = plt.subplots(figsize=(10, 6))
        ys = [_safe_float(row.get('train_raw_loss_obj_ll')) for row in self.train_epoch_rows if row.get('epoch') is not None]
        xs = [epoch for epoch, value in zip(epochs, ys) if value is not None]
        ys = [value for value in ys if value is not None]
        if xs:
            axis.plot(xs, ys, marker='o', linewidth=2.2, color=PALETTE['cyan'], label='matched_objectness')
            axis.set_xlabel('Epoch')
            axis.set_ylabel('Raw loss')
            axis.set_title('Matched Objectness Loss Component')
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False)
            self._save(figure, 'training_matched_objectness_loss_component', split='train')
        else:
            plt.close(figure)

        figure, axis = plt.subplots(figsize=(11, 6))
        plotted = False
        for label, key, color in [
            ('matched_known_knownness', 'train_raw_loss_unk_known', PALETTE['orange']),
            ('pseudo_positive_objectness', 'train_raw_loss_obj_pseudo', PALETTE['blue']),
            ('pseudo_unknown_knownness', 'train_raw_loss_unk_pseudo', PALETTE['magenta']),
            ('branch_decorrelation', 'train_raw_loss_decorr', PALETTE['green']),
        ]:
            ys = [_safe_float(row.get(key)) for row in self.train_epoch_rows if row.get('epoch') is not None]
            xs = [epoch for epoch, value in zip(epochs, ys) if value is not None]
            ys = [value for value in ys if value is not None]
            if xs:
                plotted = True
                axis.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
        if plotted:
            axis.set_xlabel('Epoch')
            axis.set_ylabel('Raw loss')
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
        plotted = False
        for label, key, color in [
            ('selection_ratio', 'pseudo_positive_selection_ratio', PALETTE['cyan']),
            ('accept_ratio', 'pseudo_positive_accept_ratio', PALETTE['red']),
        ]:
            ys = [_safe_float(row.get(key)) for row in self.train_epoch_rows if row.get('epoch') is not None]
            xs = [epoch for epoch, value in zip(epochs, ys) if value is not None]
            ys = [value for value in ys if value is not None]
            if xs:
                plotted = True
                axis.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
        if plotted:
            axis.set_xlabel('Epoch')
            axis.set_ylabel('Ratio')
            axis.set_title('Pseudo Mining Efficiency')
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False)
            self._save(figure, 'pseudo_mining_efficiency', split='train')
        else:
            plt.close(figure)

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

    def refresh_step_training_plots(self):
        if not self.train_step_rows:
            return
        self._plot_step_group(
            file_stem='step_total_loss',
            title='Step-level Total Loss',
            series=[('total_loss', 'train/loss/total', PALETTE['blue'])],
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
            ylabel='Raw loss',
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

    def refresh_step_pseudo_plots(self):
        if not self.train_step_rows:
            return
        value_keys = [
            ('selected', 'train/pseudo/selected_queries', PALETTE['blue']),
            ('candidates', 'train/pseudo/candidate_queries', PALETTE['green']),
            ('reliable_bg', 'train/pseudo/reliable_background_queries', PALETTE['orange']),
            ('ignored', 'train/pseudo/ignored_queries', PALETTE['magenta']),
        ]
        aggregated = self._aggregate_step_values([key for _, key, _ in value_keys], window_size=self.step_bar_interval)
        if aggregated is None:
            return
        xs, data = aggregated
        if len(xs) == 0:
            return
        figure, axis = plt.subplots(figsize=(13, 6.5))
        bar_width = self.step_bar_interval * 0.18
        offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, num=len(value_keys))
        for offset, (label, key, color) in zip(offsets, value_keys):
            ys = data[key]
            axis.bar(np.asarray(xs) + offset, ys, width=bar_width, color=color, alpha=0.9, label=label)
        axis.set_xlabel(f'Global step (window={self.step_bar_interval})')
        axis.set_ylabel('Average count per step')
        axis.set_title('Pseudo Mining Counts Aggregated by Step Window')
        axis.grid(True, axis='y', alpha=0.25)
        axis.legend(frameon=False, ncol=2)
        self._save(figure, 'step_pseudo_mining_counts_bars', split='train')

    def refresh_step_auxiliary_plots(self):
        if not self.train_step_rows:
            return
        self._plot_step_auxiliary_family(
            prefixes=['train/loss_raw/loss_obj_pseudo_'],
            file_stem='step_aux_obj_pseudo_loss_trends',
            title='Step-level Auxiliary Objectness Pseudo Loss Trends',
        )
        self._plot_step_auxiliary_family(
            prefixes=['train/loss_raw/loss_unk_pseudo_'],
            file_stem='step_aux_unk_pseudo_loss_trends',
            title='Step-level Auxiliary Unknownness Pseudo Loss Trends',
        )
        self._plot_step_auxiliary_family(
            prefixes=['train/loss_raw/loss_decorr_'],
            file_stem='step_aux_decorr_loss_trends',
            title='Step-level Auxiliary Decorrelation Loss Trends',
        )

    def _plot_step_auxiliary_family(self, prefixes, file_stem, title):
        keys = sorted({
            key for row in self.train_step_rows for key in row.keys()
            if any(key.startswith(prefix) for prefix in prefixes)
        })
        if not keys:
            return
        figure, axis = plt.subplots(figsize=(12, 6.5))
        colors = [PALETTE['blue'], PALETTE['orange'], PALETTE['green'], PALETTE['magenta'], PALETTE['cyan'], PALETTE['red'], PALETTE['purple']]
        plotted = False
        for index, key in enumerate(keys):
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
            axis.set_ylabel('Raw loss')
            axis.set_title(title)
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize=8, ncol=2)
            self._save(figure, file_stem, split='train')
        else:
            plt.close(figure)

    def _aggregate_step_values(self, keys, window_size=1000):
        valid_rows = [row for row in self.train_step_rows if row.get('global_step') is not None]
        if not valid_rows:
            return None
        valid_rows = sorted(valid_rows, key=lambda item: int(item['global_step']))
        grouped = {}
        counts = {}
        for row in valid_rows:
            step = int(row['global_step'])
            bucket_end = ((step // window_size) + 1) * window_size
            if bucket_end not in grouped:
                grouped[bucket_end] = {key: 0.0 for key in keys}
                counts[bucket_end] = {key: 0 for key in keys}
            for key in keys:
                value = _safe_float(row.get(key))
                if value is None:
                    continue
                grouped[bucket_end][key] += value
                counts[bucket_end][key] += 1
        xs = sorted(grouped.keys())
        data = {}
        for key in keys:
            data[key] = []
            for bucket in xs:
                count = counts[bucket][key]
                data[key].append(grouped[bucket][key] / max(count, 1))
        return xs, data

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
