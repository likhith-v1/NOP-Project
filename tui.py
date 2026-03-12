#!/usr/bin/env python3
"""
Terminal UI for visualizing optimizer comparison results.
Uses the Textual library with plotext charts.

Run:  python tui.py
"""

import json
import math
from pathlib import Path

import plotext as plt

from textual import on
from textual.app import App
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    OptionList,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from textual_plotext import PlotextPlot

# ── Constants ────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"

DATASET_META = {
    "chest_xray": {
        "label": "Chest X-Ray (Binary)",
        "deprecated": True,
    },
    "retinal_oct": {
        "label": "Retinal OCT (4-Class)",
        "deprecated": False,
    },
}


def get_dataset_label(dataset_key):
    dataset = DATASET_META[dataset_key]
    suffix = " [Deprecated]" if dataset["deprecated"] else ""
    return f"{dataset['label']}{suffix}"


DATASETS = {key: get_dataset_label(key) for key in DATASET_META}

OPTIMIZERS = ["lipschitz_momentum", "heavy_ball", "nesterov", "adam"]

OPT_LABELS = {
    "lipschitz_momentum": "LBM (Ours)",
    "heavy_ball": "Heavy-Ball",
    "nesterov": "Nesterov",
    "adam": "Adam",
}

OPT_COLORS = {
    "lipschitz_momentum": "red",
    "heavy_ball": "blue",
    "nesterov": "cyan",
    "adam": "yellow",
}

OPT_MARKERS = {
    "lipschitz_momentum": "braille",
    "heavy_ball": "braille",
    "nesterov": "braille",
    "adam": "braille",
}

METRIC_KEYS = ["accuracy", "recall", "precision", "f1", "auc_roc", "auprc"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1 Score",
    "auc_roc": "AUC-ROC",
    "auprc": "AUPRC",
}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_data():
    """Load all logs and test results for both datasets."""
    data = {}
    for ds_key in DATASETS:
        data[ds_key] = {"logs": {}, "test_results": {}}
        log_dir = RESULTS_DIR / ds_key / "logs"
        plot_dir = RESULTS_DIR / ds_key / "plots"

        for opt in OPTIMIZERS:
            log_path = log_dir / f"{opt}.json"
            if log_path.exists():
                with open(log_path) as f:
                    data[ds_key]["logs"][opt] = json.load(f)

            test_path = plot_dir / opt / "test_results.json"
            if test_path.exists():
                with open(test_path) as f:
                    data[ds_key]["test_results"][opt] = json.load(f)

    return data


# ── Widgets ──────────────────────────────────────────────────────────────────

class OverviewTable(Static):
    """Displays a comparison table of test metrics across all optimizers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dataset = "retinal_oct"
        self._data = {}

    def set_data(self, data, dataset):
        self._data = data
        self._dataset = dataset
        self.refresh_table()

    def refresh_table(self):
        test_results = self._data.get(self._dataset, {}).get("test_results", {})
        if not test_results:
            self.update("[dim]No test results found.[/dim]")
            return

        header = f"{'Metric':<14}"
        for opt in OPTIMIZERS:
            if opt in test_results:
                header += f"  {OPT_LABELS[opt]:>14}"
        lines = [header, "─" * len(header)]

        # Test loss row
        row = f"{'Test Loss':<14}"
        for opt in OPTIMIZERS:
            if opt in test_results:
                val = test_results[opt].get("test_loss", 0)
                row += f"  {val:>14.4f}"
        lines.append(row)

        # Checkpoint epoch row
        row = f"{'Best Epoch':<14}"
        for opt in OPTIMIZERS:
            if opt in test_results:
                val = test_results[opt].get("checkpoint_epoch", "?")
                row += f"  {val:>14}"
        lines.append(row)

        lines.append("─" * len(header))

        # Metric rows
        for mk in METRIC_KEYS:
            row = f"{METRIC_LABELS[mk]:<14}"
            vals = []
            for opt in OPTIMIZERS:
                if opt in test_results:
                    v = test_results[opt]["metrics"].get(mk, 0)
                    vals.append((opt, v))

            best_val = max(v for _, v in vals) if vals else 0
            for opt, v in vals:
                if abs(v - best_val) < 1e-6:
                    row += f"  [bold green]{v:>14.4f}[/bold green]"
                else:
                    row += f"  {v:>14.4f}"
            lines.append(row)

        self.update("\n".join(lines))


class ConfusionMatrixDisplay(Static):
    """Displays confusion matrices for all optimizers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dataset = "retinal_oct"
        self._data = {}

    def set_data(self, data, dataset):
        self._data = data
        self._dataset = dataset
        self.refresh_display()

    def refresh_display(self):
        test_results = self._data.get(self._dataset, {}).get("test_results", {})
        if not test_results:
            self.update("[dim]No test results found.[/dim]")
            return

        is_binary = self._dataset == "chest_xray"
        class_names = ["NORMAL", "PNEUMONIA"] if is_binary else ["CNV", "DME", "DRUSEN", "NORMAL"]

        sections = []
        for opt in OPTIMIZERS:
            if opt not in test_results:
                continue
            tr = test_results[opt]
            cm = tr.get("confusion_matrix", [])
            if not cm:
                continue

            section = f"[bold]{OPT_LABELS[opt]}[/bold]  (epoch {tr.get('checkpoint_epoch', '?')})\n"

            # Header
            header = f"  {'':>8}"
            for cn in class_names:
                header += f"  {cn:>8}"
            section += header + "\n"

            for i, row_vals in enumerate(cm):
                row = f"  {class_names[i]:>8}"
                for j, v in enumerate(row_vals):
                    if i == j:
                        row += f"  [bold green]{v:>8}[/bold green]"
                    elif v > 0:
                        row += f"  [bold red]{v:>8}[/bold red]"
                    else:
                        row += f"  {v:>8}"
                section += row + "\n"

            # Per-class metrics for retinal_oct
            if not is_binary and "per_class" in tr:
                section += "\n  Per-Class Metrics:\n"
                section += f"  {'Class':>8}  {'Recall':>8}  {'Prec':>8}  {'F1':>8}\n"
                for cn in class_names:
                    pc = tr["per_class"].get(cn, {})
                    r = pc.get("recall", 0)
                    p = pc.get("precision", 0)
                    f = pc.get("f1", 0)
                    section += f"  {cn:>8}  {r:>8.4f}  {p:>8.4f}  {f:>8.4f}\n"

            sections.append(section)

        self.update("\n".join(sections))


class LossPlot(PlotextPlot):
    """Training and validation loss curves."""

    def __init__(self, loss_type="train", **kwargs):
        super().__init__(**kwargs)
        self._loss_type = loss_type
        self._dataset = "retinal_oct"
        self._data = {}

    def set_data(self, data, dataset):
        self._data = data
        self._dataset = dataset
        self.refresh_plot()

    def refresh_plot(self):
        self.plt.clear_data()
        self.plt.clear_figure()

        logs = self._data.get(self._dataset, {}).get("logs", {})
        key = "train_loss" if self._loss_type == "train" else "val_loss"
        title = "Training Loss" if self._loss_type == "train" else "Validation Loss"

        for opt in OPTIMIZERS:
            if opt not in logs:
                continue
            history = logs[opt]["history"]
            epochs = history["epoch"]
            values = history[key]
            self.plt.plot(
                epochs, values,
                label=OPT_LABELS[opt],
                color=OPT_COLORS[opt],
                marker=OPT_MARKERS[opt],
            )

        self.plt.title(f"{title} — {DATASETS[self._dataset]}")
        self.plt.xlabel("Epoch")
        self.plt.ylabel("Loss")
        self.refresh()


class MetricPlot(PlotextPlot):
    """Metric curves over epochs."""

    def __init__(self, metric_key="recall", **kwargs):
        super().__init__(**kwargs)
        self._metric_key = metric_key
        self._dataset = "retinal_oct"
        self._data = {}

    def set_data(self, data, dataset, metric_key=None):
        self._data = data
        self._dataset = dataset
        if metric_key:
            self._metric_key = metric_key
        self.refresh_plot()

    def refresh_plot(self):
        self.plt.clear_data()
        self.plt.clear_figure()

        logs = self._data.get(self._dataset, {}).get("logs", {})

        for opt in OPTIMIZERS:
            if opt not in logs:
                continue
            history = logs[opt]["history"]
            epochs = history["epoch"]
            values = history.get(self._metric_key, [])
            if not values:
                continue
            self.plt.plot(
                epochs, values,
                label=OPT_LABELS[opt],
                color=OPT_COLORS[opt],
                marker=OPT_MARKERS[opt],
            )

        self.plt.title(f"{METRIC_LABELS.get(self._metric_key, self._metric_key)} — {DATASETS[self._dataset]}")
        self.plt.xlabel("Epoch")
        self.plt.ylabel(METRIC_LABELS.get(self._metric_key, self._metric_key))
        self.plt.ylim(0, 1.05)
        self.refresh()


class BetaPlot(PlotextPlot):
    """Beta trajectory for LBM optimizer (per-batch within each epoch)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dataset = "retinal_oct"
        self._data = {}
        self._epoch_idx = 0

    def set_data(self, data, dataset, epoch_idx=None):
        self._data = data
        self._dataset = dataset
        if epoch_idx is not None:
            self._epoch_idx = epoch_idx
        self.refresh_plot()

    def refresh_plot(self):
        self.plt.clear_data()
        self.plt.clear_figure()

        logs = self._data.get(self._dataset, {}).get("logs", {})
        lbm = logs.get("lipschitz_momentum", {})
        beta_traj = lbm.get("beta_trajectory", [])

        if not beta_traj:
            self.plt.title("No β trajectory data available")
            self.refresh()
            return

        n_epochs = len(beta_traj)
        self._epoch_idx = max(0, min(self._epoch_idx, n_epochs - 1))

        # Show the selected epoch's per-batch beta values
        betas = beta_traj[self._epoch_idx]
        steps = list(range(1, len(betas) + 1))

        self.plt.plot(steps, betas, color="red", marker="braille", label=f"Epoch {self._epoch_idx + 1}")
        self.plt.title(f"β Trajectory — Epoch {self._epoch_idx + 1}/{n_epochs} — {DATASETS[self._dataset]}")
        self.plt.xlabel("Batch Step")
        self.plt.ylabel("β_t")
        self.plt.ylim(0.84, 1.0)
        self.refresh()


class LipschitzPlot(PlotextPlot):
    """Lipschitz constant trajectory for LBM optimizer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dataset = "retinal_oct"
        self._data = {}
        self._epoch_idx = 0

    def set_data(self, data, dataset, epoch_idx=None):
        self._data = data
        self._dataset = dataset
        if epoch_idx is not None:
            self._epoch_idx = epoch_idx
        self.refresh_plot()

    def refresh_plot(self):
        self.plt.clear_data()
        self.plt.clear_figure()

        logs = self._data.get(self._dataset, {}).get("logs", {})
        lbm = logs.get("lipschitz_momentum", {})
        lip_traj = lbm.get("lipschitz_trajectory", [])

        if not lip_traj:
            self.plt.title("No Lipschitz trajectory data available")
            self.refresh()
            return

        n_epochs = len(lip_traj)
        self._epoch_idx = max(0, min(self._epoch_idx, n_epochs - 1))

        lip_vals = lip_traj[self._epoch_idx]
        steps = list(range(1, len(lip_vals) + 1))

        self.plt.plot(steps, lip_vals, color="cyan", marker="braille", label=f"Epoch {self._epoch_idx + 1}")
        self.plt.title(f"Lipschitz (L_t) — Epoch {self._epoch_idx + 1}/{n_epochs} — {DATASETS[self._dataset]}")
        self.plt.xlabel("Batch Step")
        self.plt.ylabel("L_t")
        self.refresh()


class BetaEpochAveragePlot(PlotextPlot):
    """Average beta per epoch across all batches."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dataset = "retinal_oct"
        self._data = {}

    def set_data(self, data, dataset):
        self._data = data
        self._dataset = dataset
        self.refresh_plot()

    def refresh_plot(self):
        self.plt.clear_data()
        self.plt.clear_figure()

        logs = self._data.get(self._dataset, {}).get("logs", {})
        lbm = logs.get("lipschitz_momentum", {})
        beta_traj = lbm.get("beta_trajectory", [])

        if not beta_traj:
            self.plt.title("No β trajectory data available")
            self.refresh()
            return

        avg_betas = [sum(epoch_betas) / len(epoch_betas) for epoch_betas in beta_traj]
        min_betas = [min(epoch_betas) for epoch_betas in beta_traj]
        max_betas = [max(epoch_betas) for epoch_betas in beta_traj]
        epochs = list(range(1, len(avg_betas) + 1))

        self.plt.plot(epochs, avg_betas, color="red", marker="braille", label="Mean β")
        self.plt.plot(epochs, min_betas, color="yellow", marker="braille", label="Min β")
        self.plt.plot(epochs, max_betas, color="green", marker="braille", label="Max β")

        self.plt.title(f"β Summary Per Epoch — {DATASETS[self._dataset]}")
        self.plt.xlabel("Epoch")
        self.plt.ylabel("β_t")
        self.plt.ylim(0.84, 1.0)
        self.refresh()


class LipschitzEpochAveragePlot(PlotextPlot):
    """Average Lipschitz constant per epoch."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dataset = "retinal_oct"
        self._data = {}

    def set_data(self, data, dataset):
        self._data = data
        self._dataset = dataset
        self.refresh_plot()

    def refresh_plot(self):
        self.plt.clear_data()
        self.plt.clear_figure()

        logs = self._data.get(self._dataset, {}).get("logs", {})
        lbm = logs.get("lipschitz_momentum", {})
        lip_traj = lbm.get("lipschitz_trajectory", [])

        if not lip_traj:
            self.plt.title("No Lipschitz trajectory data available")
            self.refresh()
            return

        avg_lips = [sum(el) / len(el) for el in lip_traj]
        median_lips = [sorted(el)[len(el) // 2] for el in lip_traj]
        epochs = list(range(1, len(avg_lips) + 1))

        self.plt.plot(epochs, avg_lips, color="cyan", marker="braille", label="Mean L")
        self.plt.plot(epochs, median_lips, color="magenta", marker="braille", label="Median L")

        self.plt.title(f"Lipschitz (L_t) Summary Per Epoch — {DATASETS[self._dataset]}")
        self.plt.xlabel("Epoch")
        self.plt.ylabel("L_t")
        self.refresh()


# ── Main App ─────────────────────────────────────────────────────────────────

class OptimizerDashboard(App):
    """Terminal dashboard for visualizing optimizer comparison results."""

    CSS = """
    Screen {
        background: $surface;
    }

    #sidebar {
        width: 28;
        dock: left;
        background: $panel;
        padding: 1 1;
        border-right: tall $accent;
    }

    #sidebar Label {
        margin: 1 0 0 0;
        text-style: bold;
        color: $text;
    }

    #main-content {
        padding: 0 1;
    }

    .plot-container {
        height: 1fr;
    }

    .plot-row {
        height: 1fr;
    }

    PlotextPlot {
        height: 1fr;
        min-height: 14;
    }

    #overview-table {
        padding: 1 2;
        height: auto;
    }

    #confusion-display {
        padding: 1 2;
        height: auto;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        margin: 1 0;
        text-align: center;
    }

    #epoch-label {
        text-align: center;
        color: $text-muted;
        margin: 0 0;
    }

    .summary-row {
        height: 1fr;
        min-height: 14;
    }

    .detail-row {
        height: 1fr;
        min-height: 14;
    }
    """

    BINDINGS = [
        Binding("d", "switch_dataset", "Switch Dataset"),
        Binding("right", "next_epoch", "Next Epoch", show=True),
        Binding("left", "prev_epoch", "Prev Epoch", show=True),
        Binding("q", "quit", "Quit"),
    ]

    TITLE = "Optimizer Comparison Dashboard"
    SUB_TITLE = "NOP-Project · LBM vs Heavy-Ball vs Nesterov vs Adam"

    current_dataset = reactive("retinal_oct")
    current_epoch_idx = reactive(0)

    def __init__(self):
        super().__init__()
        self._data = load_all_data()

    def compose(self):
        yield Header()

        with Horizontal():
            # Sidebar
            with Vertical(id="sidebar"):
                yield Label("📊 Dataset")
                yield Select(
                    [(v, k) for k, v in DATASETS.items()],
                    value="retinal_oct",
                    id="dataset-select",
                    allow_blank=False,
                )
                yield Label("📈 Metric")
                yield Select(
                    [(v, k) for k, v in METRIC_LABELS.items()],
                    value="recall",
                    id="metric-select",
                    allow_blank=False,
                )
                yield Label("")
                yield Static(
                    "[bold]Keybindings[/bold]\n"
                    "─────────────\n"
                    "[bold cyan]d[/bold cyan]  Switch dataset\n"
                    "[bold cyan]←→[/bold cyan] Change epoch\n"
                    "[bold cyan]q[/bold cyan]  Quit\n",
                    id="help-text",
                )
                yield Label("")
                yield Static(id="dataset-info")

            # Main content area
            with Vertical(id="main-content"):
                with TabbedContent(
                    "Overview",
                    "Loss Curves",
                    "Metrics",
                    "β & L Trajectories",
                    "Confusion Matrices",
                ):
                    # Tab 1: Overview
                    with TabPane("Overview"):
                        yield Static("[bold]Test Metrics Comparison[/bold]", classes="section-title")
                        with VerticalScroll():
                            yield OverviewTable(id="overview-table")

                    # Tab 2: Loss Curves
                    with TabPane("Loss Curves"):
                        with Horizontal(classes="plot-row"):
                            yield LossPlot(loss_type="train", id="train-loss-plot")
                            yield LossPlot(loss_type="val", id="val-loss-plot")

                    # Tab 3: Metrics
                    with TabPane("Metrics"):
                        with Horizontal(classes="plot-row"):
                            yield MetricPlot(metric_key="recall", id="metric-plot-1")
                            yield MetricPlot(metric_key="accuracy", id="metric-plot-2")
                        with Horizontal(classes="plot-row"):
                            yield MetricPlot(metric_key="f1", id="metric-plot-3")
                            yield MetricPlot(metric_key="auc_roc", id="metric-plot-4")

                    # Tab 4: β & L Trajectories (LBM-specific)
                    with TabPane("β & L Trajectories"):
                        yield Label("", id="epoch-label")
                        with Horizontal(classes="summary-row"):
                            yield BetaEpochAveragePlot(id="beta-avg-plot")
                            yield LipschitzEpochAveragePlot(id="lip-avg-plot")
                        with Horizontal(classes="detail-row"):
                            yield BetaPlot(id="beta-plot")
                            yield LipschitzPlot(id="lip-plot")

                    # Tab 5: Confusion Matrices
                    with TabPane("Confusion Matrices"):
                        yield Static("[bold]Confusion Matrices & Per-Class Metrics[/bold]", classes="section-title")
                        with VerticalScroll():
                            yield ConfusionMatrixDisplay(id="confusion-display")

        yield Footer()

    def on_mount(self):
        self._refresh_all()

    def _refresh_all(self):
        ds = self.current_dataset
        metric_select = self.query_one("#metric-select", Select)
        metric_key = metric_select.value if metric_select.value != Select.BLANK else "recall"

        # Overview table
        self.query_one("#overview-table", OverviewTable).set_data(self._data, ds)

        # Loss plots
        self.query_one("#train-loss-plot", LossPlot).set_data(self._data, ds)
        self.query_one("#val-loss-plot", LossPlot).set_data(self._data, ds)

        # Metric plots
        self.query_one("#metric-plot-1", MetricPlot).set_data(self._data, ds, "recall")
        self.query_one("#metric-plot-2", MetricPlot).set_data(self._data, ds, "accuracy")
        self.query_one("#metric-plot-3", MetricPlot).set_data(self._data, ds, "f1")
        self.query_one("#metric-plot-4", MetricPlot).set_data(self._data, ds, "auc_roc")

        # LBM trajectory plots
        self.query_one("#beta-avg-plot", BetaEpochAveragePlot).set_data(self._data, ds)
        self.query_one("#lip-avg-plot", LipschitzEpochAveragePlot).set_data(self._data, ds)
        self.query_one("#beta-plot", BetaPlot).set_data(self._data, ds, self.current_epoch_idx)
        self.query_one("#lip-plot", LipschitzPlot).set_data(self._data, ds, self.current_epoch_idx)

        # Epoch label
        lbm_logs = self._data.get(ds, {}).get("logs", {}).get("lipschitz_momentum", {})
        n_epochs = len(lbm_logs.get("beta_trajectory", []))
        epoch_label = self.query_one("#epoch-label", Label)
        if n_epochs > 0:
            epoch_label.update(
                f"[bold]Epoch {self.current_epoch_idx + 1} / {n_epochs}[/bold]  "
                f"(Use ← → to navigate epochs)"
            )
        else:
            epoch_label.update("[dim]No trajectory data for this dataset[/dim]")

        # Confusion matrix
        self.query_one("#confusion-display", ConfusionMatrixDisplay).set_data(self._data, ds)

        # Dataset info in sidebar
        info_widget = self.query_one("#dataset-info", Static)
        logs = self._data.get(ds, {}).get("logs", {})
        test_results = self._data.get(ds, {}).get("test_results", {})
        info_lines = [
            f"[bold]Dataset Info[/bold]",
            f"─────────────",
            f"Status: {'Deprecated' if DATASET_META[ds]['deprecated'] else 'Active'}",
            f"Optimizers: {len(logs)}",
            f"Test results: {len(test_results)}",
        ]
        if DATASET_META[ds]["deprecated"]:
            info_lines.append("[yellow]Chest X-Ray is deprecated.[/yellow]")
        if logs:
            opt0 = next(iter(logs))
            n_ep = len(logs[opt0]["history"]["epoch"])
            info_lines.append(f"Max epochs: {n_ep}")
        info_widget.update("\n".join(info_lines))

    @on(Select.Changed, "#dataset-select")
    def dataset_changed(self, event):
        self.current_dataset = event.value
        self.current_epoch_idx = 0
        self._refresh_all()

    @on(Select.Changed, "#metric-select")
    def metric_changed(self, event):
        mk = event.value
        if mk == Select.BLANK:
            return
        self.query_one("#metric-plot-1", MetricPlot).set_data(self._data, self.current_dataset, mk)

    def action_switch_dataset(self):
        ds_select = self.query_one("#dataset-select", Select)
        keys = list(DATASETS.keys())
        cur_idx = keys.index(self.current_dataset)
        new_idx = (cur_idx + 1) % len(keys)
        ds_select.value = keys[new_idx]

    def action_next_epoch(self):
        lbm_logs = self._data.get(self.current_dataset, {}).get("logs", {}).get("lipschitz_momentum", {})
        n_epochs = len(lbm_logs.get("beta_trajectory", []))
        if n_epochs > 0:
            self.current_epoch_idx = (self.current_epoch_idx + 1) % n_epochs
            self.query_one("#beta-plot", BetaPlot).set_data(self._data, self.current_dataset, self.current_epoch_idx)
            self.query_one("#lip-plot", LipschitzPlot).set_data(self._data, self.current_dataset, self.current_epoch_idx)
            epoch_label = self.query_one("#epoch-label", Label)
            epoch_label.update(
                f"[bold]Epoch {self.current_epoch_idx + 1} / {n_epochs}[/bold]  "
                f"(Use ← → to navigate epochs)"
            )

    def action_prev_epoch(self):
        lbm_logs = self._data.get(self.current_dataset, {}).get("logs", {}).get("lipschitz_momentum", {})
        n_epochs = len(lbm_logs.get("beta_trajectory", []))
        if n_epochs > 0:
            self.current_epoch_idx = (self.current_epoch_idx - 1) % n_epochs
            self.query_one("#beta-plot", BetaPlot).set_data(self._data, self.current_dataset, self.current_epoch_idx)
            self.query_one("#lip-plot", LipschitzPlot).set_data(self._data, self.current_dataset, self.current_epoch_idx)
            epoch_label = self.query_one("#epoch-label", Label)
            epoch_label.update(
                f"[bold]Epoch {self.current_epoch_idx + 1} / {n_epochs}[/bold]  "
                f"(Use ← → to navigate epochs)"
            )


if __name__ == "__main__":
    app = OptimizerDashboard()
    app.run()
