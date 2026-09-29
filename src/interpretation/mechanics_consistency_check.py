from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import load_modeling_dataset
from src.models.model_selection import build_model_ranking
from src.utils.io_utils import dataframe_to_markdown
from src.utils.paths import outputs_dir


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load_best_model_for_target(target: str):
    ranking_path = outputs_dir() / "tables" / "model_comparison" / "final_model_ranking.xlsx"
    if not ranking_path.exists():
        build_model_ranking()
    ranking = pd.read_excel(ranking_path, sheet_name="final_model_ranking")
    eligible = ranking.get("eligible_for_surrogate", pd.Series(False, index=ranking.index)).fillna(False).astype(bool)
    sub = ranking[(ranking["target"] == target) & eligible].sort_values("composite_score", ascending=False).copy()
    if sub.empty:
        return None
    row = sub.iloc[0]
    registry = pd.read_csv(outputs_dir() / "models" / "model_registry.csv")
    match = registry[(registry["task"] == row["task"]) & (registry["target"] == row["target"]) & (registry["model"] == row["model"])]
    if match.empty:
        return None
    model_path = Path(match.iloc[0]["model_path"])
    metadata_path = Path(match.iloc[0]["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return joblib.load(model_path), metadata


def run_mechanics_consistency_check() -> dict[str, Path]:
    df = load_modeling_dataset().reset_index(drop=True)
    pred = df[["specimen_id", "group", "direction", "lever_arm_length", "GI_raw", "GII_raw", "Gc_raw", "GII_fraction"]].copy()
    for target in ["GI_raw", "GII_raw", "Gc_raw", "GII_fraction"]:
        loaded = _load_best_model_for_target(target)
        if loaded is None:
            continue
        model, meta = loaded
        features = meta["input_features"]
        pred[f"{target}_pred"] = model.predict(df[features])
        pred[f"{target}_model"] = meta["model_name"]
        pred[f"{target}_task"] = meta["task_name"]

    if {"GI_raw_pred", "GII_raw_pred", "Gc_raw_pred"}.issubset(pred.columns):
        pred["GI_positive_pred"] = pred["GI_raw_pred"] > 0
        pred["GII_positive_pred"] = pred["GII_raw_pred"] > 0
        pred["Gc_consistency_relative_error"] = (
            pred["Gc_raw_pred"] - pred["GI_raw_pred"] - pred["GII_raw_pred"]
        ).abs() / pred["Gc_raw_pred"].abs().replace(0, np.nan)
    else:
        pred["GI_positive_pred"] = np.nan
        pred["GII_positive_pred"] = np.nan
        pred["Gc_consistency_relative_error"] = np.nan
    pred["GII_fraction_reasonable"] = pred.get("GII_fraction_pred", pred["GII_fraction"]).between(0, 1)

    summary_rows = [
        {"check": "predicted_GI_positive_rate", "value": float(pd.to_numeric(pred["GI_positive_pred"], errors="coerce").mean(skipna=True))},
        {"check": "predicted_GII_positive_rate", "value": float(pd.to_numeric(pred["GII_positive_pred"], errors="coerce").mean(skipna=True))},
        {"check": "Gc_equals_GI_plus_GII_mean_relative_error", "value": float(pred["Gc_consistency_relative_error"].mean(skipna=True))},
        {"check": "GII_fraction_reasonable_rate", "value": float(pred["GII_fraction_reasonable"].mean(skipna=True))},
    ]
    summary = pd.DataFrame(summary_rows)
    out_metrics = outputs_dir() / "metrics" / "physics_consistency"
    out_reports = outputs_dir() / "reports"
    out_fig = outputs_dir() / "figures" / "physics_consistency"
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)
    xlsx = out_metrics / "physics_consistency_summary.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        pred.to_excel(writer, sheet_name="case_level_predictions", index=False)

    report = out_reports / "physics_consistency_report.md"
    report.write_text(
        "\n".join(
            [
                "# Physics Consistency Report",
                "",
                "Status: **BLOCKED_NO_VALIDATED_SURROGATE**" if "Gc_raw_pred" not in pred.columns else "Status: validated-surrogate checks completed.",
                "",
                "The checks below are sanity checks for data-driven surrogate predictions. They do not replace additional MMB experiments or finite-element validation.",
                "",
                dataframe_to_markdown(summary),
                "",
                "Predictions that violate positivity, mode-mixity bounds, or `Gc = GI + GII` consistency should be treated as conflict cases and discussed as small-sample surrogate limitations.",
            ]
        ),
        encoding="utf-8",
    )

    plt = _plt()
    if {"GI_raw_pred", "GII_raw_pred"}.issubset(pred.columns):
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        ax.scatter(pred["GI_raw_pred"], pred["GII_raw_pred"], c=pred["lever_arm_length"], cmap="viridis", edgecolor="black")
        ax.set_xlabel("Predicted GI")
        ax.set_ylabel("Predicted GII")
        fig.savefig(out_fig / "GI_GII_physics_consistency.png", bbox_inches="tight", dpi=300)
        fig.savefig(out_fig / "GI_GII_physics_consistency.pdf", bbox_inches="tight")
        pred.to_csv(out_fig / "GI_GII_physics_consistency_data.csv", index=False)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        ax.scatter(pred["GI_raw_pred"] + pred["GII_raw_pred"], pred["Gc_raw_pred"], edgecolor="black")
        lo = np.nanmin([pred["GI_raw_pred"].add(pred["GII_raw_pred"]).min(), pred["Gc_raw_pred"].min()])
        hi = np.nanmax([pred["GI_raw_pred"].add(pred["GII_raw_pred"]).max(), pred["Gc_raw_pred"].max()])
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--")
        ax.set_xlabel("Predicted GI + GII")
        ax.set_ylabel("Predicted Gc")
        fig.savefig(out_fig / "Gc_equals_GI_plus_GII_check.png", bbox_inches="tight", dpi=300)
        fig.savefig(out_fig / "Gc_equals_GI_plus_GII_check.pdf", bbox_inches="tight")
        pred.to_csv(out_fig / "Gc_equals_GI_plus_GII_check_data.csv", index=False)
        plt.close(fig)

    if "Gc_raw_pred" in pred:
        trend = pred.groupby(["direction", "lever_arm_length"])["Gc_raw_pred"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for direction, sub in trend.groupby("direction"):
            ax.errorbar(sub["lever_arm_length"], sub["mean"], yerr=sub["std"], marker="o", label=direction)
        ax.set_xlabel("Lever-arm length c (mm)")
        ax.set_ylabel("Predicted Gc")
        ax.legend(frameon=False)
        fig.savefig(out_fig / "predicted_trend_with_lever_arm.png", bbox_inches="tight", dpi=300)
        fig.savefig(out_fig / "predicted_trend_with_lever_arm.pdf", bbox_inches="tight")
        trend.to_csv(out_fig / "predicted_trend_with_lever_arm_data.csv", index=False)
        plt.close(fig)
    return {"summary": xlsx, "report": report}


if __name__ == "__main__":
    run_mechanics_consistency_check()
