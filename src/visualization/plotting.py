from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.compute_mmb_parameters import compute_specimen_parameters
from src.utils.paths import data_dir, outputs_dir


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    return plt


def save_figure(fig, base_path: Path, data: pd.DataFrame | None = None) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    if data is not None:
        data.to_csv(base_path.with_name(base_path.name + "_data.csv"), index=False)


def experimental_figures() -> list[Path]:
    compute_specimen_parameters()
    plt = _plt()
    point_path = data_dir() / "processed" / "point_data_all_rows.csv"
    if not point_path.exists():
        point_path = data_dir() / "interim" / "point_data_all_rows.csv"
    point = pd.read_csv(point_path)
    summary = pd.read_csv(data_dir() / "processed" / "modeling_dataset_specimen_level.csv")
    out = outputs_dir() / "figures" / "experimental"
    paths = []

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for specimen_id, sub in point.groupby("specimen_id"):
        sub = sub.sort_values("end_displacement")
        ax.plot(sub["end_displacement"], sub["load_for_analysis"], linewidth=0.8, alpha=0.55)
    ax.set_xlabel("End displacement (mm)")
    ax.set_ylabel("Load (N)")
    ax.set_title("Load-displacement curves")
    ax.grid(alpha=0.25)
    base = out / "load_displacement_curves"
    save_figure(fig, base, point[["specimen_id", "group", "end_displacement", "load_for_analysis"]])
    paths.append(base.with_suffix(".png"))
    plt.close(fig)

    plot_specs = [
        ("Pmax", "Pmax_by_lever_arm", "Peak load"),
        ("GI_raw", "GI_by_lever_arm", "GI"),
        ("GII_raw", "GII_by_lever_arm", "GII"),
        ("Gc_raw", "Gc_by_lever_arm", "Gc"),
        ("GI_GII_ratio", "GI_GII_ratio_by_lever_arm", "GI/GII"),
        ("GII_fraction", "fracture_mode_summary", "GII/(GI+GII)"),
    ]
    for col, filename, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        order = sorted(summary["lever_arm_length"].dropna().unique())
        data = [summary.loc[summary["lever_arm_length"] == c, col].dropna() for c in order]
        ax.boxplot(data, labels=[str(int(c)) for c in order], showmeans=True)
        for i, values in enumerate(data, start=1):
            ax.scatter(np.full(len(values), i), values, s=18, color="black", alpha=0.65)
        ax.set_xlabel("Lever-arm length c (mm)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        base = out / filename
        save_figure(fig, base, summary[["specimen_id", "direction", "lever_arm_length", col]])
        paths.append(base.with_suffix(".png"))
        plt.close(fig)
    return paths


def _read_metrics(strategy: str) -> pd.DataFrame:
    path = outputs_dir() / "metrics" / strategy / f"{strategy}_metrics.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_predictions(strategy: str) -> pd.DataFrame:
    path = outputs_dir() / "predictions" / strategy / f"{strategy}_predictions.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def model_comparison_figures() -> list[Path]:
    plt = _plt()
    out = outputs_dir() / "figures" / "model_comparison"
    paths = []
    for strategy, label in [
        ("repeated_random_split", "repeated_random_split"),
        ("leave_one_lever_arm_out", "leave_one_lever_arm"),
    ]:
        metrics = _read_metrics(strategy)
        if metrics.empty:
            continue
        metrics = metrics[(metrics["task"] == "task_A_pre_test") & (metrics["target"] == "Gc_raw")].copy()
        if metrics.empty:
            continue
        for metric in ["R2", "RMSE"]:
            fig, ax = plt.subplots(figsize=(7.6, 5.6))
            data = metrics.groupby("model")[metric].agg(["mean", "std"]).sort_values("mean", ascending=(metric != "R2"))
            plot_data = data.sort_values("mean", ascending=True)
            ax.barh(
                plot_data.index,
                plot_data["mean"],
                xerr=plot_data["std"],
                capsize=3,
                color="#7aa6c2",
                edgecolor="black",
            )
            ax.set_xlabel(metric)
            ax.set_ylabel("Model")
            ax.set_title("task_A_pre_test | corrected Gc")
            if metric == "R2":
                ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.grid(axis="x", alpha=0.25)
            base = out / f"model_comparison_{label}_{metric}"
            save_figure(fig, base, data.reset_index())
            paths.append(base.with_suffix(".png"))
            plt.close(fig)
    return paths


def prediction_and_residual_figures() -> list[Path]:
    plt = _plt()
    paths = []
    ranking_path = outputs_dir() / "metrics" / "model_comparison" / "final_model_ranking.csv"
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        eligible = ranking.get("eligible_for_surrogate", pd.Series(False, index=ranking.index)).fillna(False).astype(bool)
        if not eligible.any():
            message = (
                "# Prediction and Residual Figure Status\n\n"
                "Status: **BLOCKED_NO_VALIDATED_SURROGATE**\n\n"
                "No best-model prediction or residual figure is generated because no model passed the leave-one-lever-arm generalization gate.\n"
            )
            for folder in ["predictions", "residuals"]:
                out = outputs_dir() / "figures" / folder
                out.mkdir(parents=True, exist_ok=True)
                (out / "figure_status.md").write_text(message, encoding="utf-8")
            return []
    for strategy, suffix in [
        ("repeated_random_split", "random_split"),
        ("leave_one_lever_arm_out", "leave_one_lever_arm"),
    ]:
        pred = _read_predictions(strategy)
        if pred.empty:
            continue
        metrics = _read_metrics(strategy)
        if metrics.empty:
            continue
        best = (
            metrics.groupby(["model", "task", "target"])["RMSE"]
            .mean()
            .sort_values()
            .reset_index()
            .iloc[0]
        )
        sub = pred[(pred["model"] == best["model"]) & (pred["task"] == best["task"]) & (pred["target"] == best["target"])]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        ax.scatter(sub["actual"], sub["predicted"], c=sub["lever_arm_length"], cmap="viridis", edgecolor="black")
        lo = min(sub["actual"].min(), sub["predicted"].min())
        hi = max(sub["actual"].max(), sub["predicted"].max())
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{best['model']} | {best['task']} | {best['target']}")
        base = outputs_dir() / "figures" / "predictions" / f"predicted_vs_measured_{suffix}_best_model"
        save_figure(fig, base, sub)
        paths.append(base.with_suffix(".png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.scatter(sub["predicted"], sub["residual"], c=sub["lever_arm_length"], cmap="viridis", edgecolor="black")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        base = outputs_dir() / "figures" / "residuals" / f"residuals_{suffix}_best_model"
        save_figure(fig, base, sub)
        paths.append(base.with_suffix(".png"))
        plt.close(fig)

    pred = pd.concat([_read_predictions("repeated_random_split"), _read_predictions("leave_one_lever_arm_out")], ignore_index=True)
    if not pred.empty:
        for kind, folder in [("predicted_vs_measured_by_lever_arm", "predictions"), ("residuals_by_lever_arm", "residuals")]:
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            if kind.startswith("predicted"):
                ax.scatter(pred["actual"], pred["predicted"], c=pred["lever_arm_length"], cmap="viridis", alpha=0.6)
                ax.set_xlabel("Measured")
                ax.set_ylabel("Predicted")
            else:
                ax.scatter(pred["lever_arm_length"], pred["residual"], alpha=0.6)
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.set_xlabel("Lever-arm length c (mm)")
                ax.set_ylabel("Residual")
            base = outputs_dir() / "figures" / folder / kind
            save_figure(fig, base, pred)
            paths.append(base.with_suffix(".png"))
            plt.close(fig)
    return paths


def validation_figures() -> list[Path]:
    return model_comparison_figures() + prediction_and_residual_figures()
