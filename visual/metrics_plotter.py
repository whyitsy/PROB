import json
from pathlib import Path

from util.visual.evaluation import (
    plot_branch_correlation_trends,
    plot_open_world_error_metrics,
    plot_open_world_percentage_metrics,
)
from util.visual.training import (
    append_json_record,
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
        plot_training_total_loss(self.train_epoch_rows, self.train_plots_dir / 'training_total_loss.svg')
        plot_training_base_loss_components(self.train_epoch_rows, self.train_plots_dir / 'training_base_loss_components.svg')
        plot_training_matched_objectness_loss_component(self.train_epoch_rows, self.train_plots_dir / 'training_matched_objectness_loss_component.svg')
        plot_training_open_world_loss_components(self.train_epoch_rows, self.train_plots_dir / 'training_open_world_loss_components.svg')

    def refresh_epoch_pseudo_plots(self):
        if not self.train_epoch_rows:
            return
        plot_pseudo_mining_counts(self.train_epoch_rows, self.train_plots_dir / 'pseudo_mining_counts.svg')
        plot_pseudo_mining_efficiency(self.train_epoch_rows, self.train_plots_dir / 'pseudo_mining_efficiency.svg')

    def refresh_step_training_plots(self):
        if not self.train_step_rows:
            return
        plot_step_total_loss(self.train_step_rows, self.train_plots_dir / 'step_total_loss.svg')
        plot_step_base_losses(self.train_step_rows, self.train_plots_dir / 'step_base_losses.svg')
        plot_step_open_world_losses(self.train_step_rows, self.train_plots_dir / 'step_open_world_losses.svg')
        plot_step_query_score_statistics(self.train_step_rows, self.train_plots_dir / 'step_query_score_statistics.svg')

    def refresh_step_pseudo_plots(self):
        if not self.train_step_rows:
            return
        plot_step_pseudo_mining_statistics(self.train_step_rows, self.train_plots_dir / 'step_pseudo_mining_statistics.svg')
        plot_step_pseudo_mining_counts_bars(self.train_step_rows, self.train_plots_dir / 'step_pseudo_mining_counts_bars.svg')

    def refresh_step_auxiliary_plots(self):
        if not self.train_step_rows:
            return
        plot_step_auxiliary_loss_trends(self.train_step_rows, self.train_plots_dir / 'step_auxiliary_loss_trends.svg')


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
