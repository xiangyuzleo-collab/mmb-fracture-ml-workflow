from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io_utils import dataframe_to_markdown, safe_name
from src.utils.logging_utils import get_logger
from src.utils.paths import outputs_dir


LOGGER = get_logger("model_selection")


def _read_metrics() -> pd.DataFrame:
    frames = []
    for strategy in ["repeated_random_split", "leave_one_lever_arm_out"]:
        path = outputs_dir() / "metrics" / strategy / f"{strategy}_metrics.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _norm(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.max(skipna=True) == s.min(skipna=True):
        return pd.Series(0.5, index=series.index)
    z = (s - s.min(skipna=True)) / (s.max(skipna=True) - s.min(skipna=True))
    return z if higher_is_better else 1.0 - z


def _norm_within_task_target(frame: pd.DataFrame, column: str, higher_is_better: bool) -> pd.Series:
    """Normalize metrics only among models solving the same task and target."""
    return frame.groupby(["task", "target"], group_keys=False)[column].transform(
        lambda values: _norm(values, higher_is_better=higher_is_better)
    )


def build_model_ranking() -> dict[str, Path]:
    metrics = _read_metrics()
    out_table = outputs_dir() / "tables" / "model_comparison" / "final_model_ranking.xlsx"
    out_metrics_dir = outputs_dir() / "metrics" / "model_comparison"
    out_report = outputs_dir() / "reports" / "model_selection_report.md"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        pd.DataFrame().to_excel(out_table, index=False)
        pd.DataFrame().to_csv(out_metrics_dir / "model_comparison_summary.csv", index=False)
        out_report.write_text("# Model Selection Report\n\nNo validation metrics were found.\n", encoding="utf-8")
        return {"ranking": out_table, "report": out_report}

    grouped = (
        metrics.groupby(["task", "target", "model", "model_family", "validation_strategy"])
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAPE_mean=("MAPE", "mean"),
            MaxRE_max=("MaxRE", "max"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["task", "target", "model", "model_family"],
        columns="validation_strategy",
        values=["R2_mean", "R2_std", "RMSE_mean", "RMSE_std", "MAE_mean", "MAPE_mean", "MaxRE_max", "n_folds"],
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    ranking = pivot.reset_index()
    for col in [
        "R2_mean_repeated_random_split",
        "R2_mean_leave_one_lever_arm_out",
        "RMSE_mean_repeated_random_split",
        "RMSE_mean_leave_one_lever_arm_out",
        "R2_std_repeated_random_split",
        "MaxRE_max_leave_one_lever_arm_out",
    ]:
        if col not in ranking:
            ranking[col] = np.nan
    ranking["_r2_std_for_score"] = ranking.groupby(["task", "target"])["R2_std_repeated_random_split"].transform(
        lambda values: values.fillna(values.max(skipna=True))
    )
    ranking["_rmse_for_score"] = ranking["RMSE_mean_leave_one_lever_arm_out"].fillna(
        ranking["RMSE_mean_repeated_random_split"]
    )
    ranking["_maxre_for_score"] = ranking.groupby(["task", "target"])["MaxRE_max_leave_one_lever_arm_out"].transform(
        lambda values: values.fillna(values.max(skipna=True))
    )
    ranking["score_random_interpolation"] = _norm_within_task_target(ranking, "R2_mean_repeated_random_split", True)
    ranking["score_random_stability"] = _norm_within_task_target(ranking, "_r2_std_for_score", False)
    ranking["score_leave_one_generalization"] = _norm_within_task_target(ranking, "R2_mean_leave_one_lever_arm_out", True)
    ranking["score_rmse"] = _norm_within_task_target(ranking, "_rmse_for_score", False)
    ranking["score_max_error"] = _norm_within_task_target(ranking, "_maxre_for_score", False)
    small_sample_bonus = ranking["model"].isin(["ridge_regression", "gaussian_process", "random_forest", "extra_trees"]).astype(float) * 0.05
    ranking["composite_score"] = (
        0.20 * ranking["score_random_interpolation"]
        + 0.15 * ranking["score_random_stability"]
        + 0.30 * ranking["score_leave_one_generalization"]
        + 0.15 * ranking["score_rmse"]
        + 0.15 * ranking["score_max_error"]
        + small_sample_bonus
    )
    mean_baseline = ranking[ranking["model"] == "mean_predictor"][
        [
            "task",
            "target",
            "R2_mean_leave_one_lever_arm_out",
            "RMSE_mean_leave_one_lever_arm_out",
        ]
    ].rename(
        columns={
            "R2_mean_leave_one_lever_arm_out": "mean_baseline_R2_leave_one",
            "RMSE_mean_leave_one_lever_arm_out": "mean_baseline_RMSE_leave_one",
        }
    )
    ranking = ranking.merge(mean_baseline, on=["task", "target"], how="left", validate="many_to_one")
    ranking["beats_mean_baseline_leave_one"] = (
        (ranking["R2_mean_leave_one_lever_arm_out"] > ranking["mean_baseline_R2_leave_one"])
        & (ranking["RMSE_mean_leave_one_lever_arm_out"] < ranking["mean_baseline_RMSE_leave_one"])
    )
    ranking["eligible_for_surrogate"] = (
        ranking["R2_mean_leave_one_lever_arm_out"].notna()
        & (ranking["R2_mean_leave_one_lever_arm_out"] > 0.0)
        & ranking["beats_mean_baseline_leave_one"]
    )
    ranking["selection_status"] = np.where(
        ranking["eligible_for_surrogate"],
        "eligible_after_generalization_gate",
        "screening_only_failed_generalization_gate",
    )
    ranking = ranking.sort_values(["task", "target", "composite_score"], ascending=[True, True, False])
    ranking["rank_within_task_target"] = ranking.groupby(["task", "target"])["composite_score"].rank(ascending=False, method="first")

    with pd.ExcelWriter(out_table) as writer:
        ranking.to_excel(writer, sheet_name="final_model_ranking", index=False)
        grouped.to_excel(writer, sheet_name="metric_summary_long", index=False)
    ranking.to_csv(out_metrics_dir / "final_model_ranking.csv", index=False)
    grouped.to_csv(out_metrics_dir / "model_comparison_summary.csv", index=False)

    screening_best = ranking[ranking["rank_within_task_target"] == 1].copy()
    eligible_best = screening_best[screening_best["eligible_for_surrogate"]].copy()
    selected_manifest = _export_selected_best_models(eligible_best)
    lines = [
        "# Model Selection Report",
        "",
        "Model selection is not based only on the highest repeated random split R2. The composite score combines interpolation performance, repeated-split stability, leave-one-lever-arm generalization, RMSE, maximum relative error, and a small-sample suitability bonus.",
        "",
        "If a model performs well under repeated random split but poorly under leave-one-lever-arm validation, it is interpreted as mainly interpolation-capable rather than reliably generalizable to unseen MMB loading configurations.",
        "",
        f"Models passing the generalization eligibility gate: {int(ranking['eligible_for_surrogate'].sum())}.",
        "A model is eligible only when mean leave-one-lever-arm R2 is positive and it outperforms the direction-agnostic mean baseline in both R2 and RMSE.",
        "",
        "## Top screening model by task and target",
        "",
        dataframe_to_markdown(
            screening_best[
            [
                "task",
                "target",
                "model",
                "R2_mean_repeated_random_split",
                "R2_mean_leave_one_lever_arm_out",
                "RMSE_mean_leave_one_lever_arm_out",
                "composite_score",
                "eligible_for_surrogate",
                "selection_status",
            ]
            ]
        ),
        "",
        "No model is exported as a validated surrogate when the eligibility gate is not passed. Screening winners may still be inspected for error analysis, but they must not drive inverse optimization.",
        "",
        f"Full ranking table: `{out_table}`",
        f"Selected best-model manifest: `{selected_manifest}`",
    ]
    out_report.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Model ranking written: %s", out_table)
    return {"ranking": out_table, "report": out_report, "selected_best_model_manifest": selected_manifest}


def _export_selected_best_models(best: pd.DataFrame) -> Path:
    selected_dir = outputs_dir() / "models" / "selected_best_model"
    selected_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = selected_dir / "selected_best_model_manifest.csv"
    registry_path = outputs_dir() / "models" / "model_registry.csv"
    manifest_columns = [
        "task",
        "target",
        "model",
        "model_family",
        "composite_score",
        "source_model_path",
        "source_metadata_path",
        "selected_model_path",
        "selected_metadata_path",
        "status",
    ]
    rows = []
    if best.empty or not registry_path.exists():
        pd.DataFrame(rows, columns=manifest_columns).to_csv(manifest_path, index=False)
        return manifest_path

    registry = pd.read_csv(registry_path)
    for _, row in best.iterrows():
        match = registry[
            (registry["task"] == row["task"])
            & (registry["target"] == row["target"])
            & (registry["model"] == row["model"])
        ].copy()
        if match.empty:
            rows.append(
                {
                    "task": row.get("task", ""),
                    "target": row.get("target", ""),
                    "model": row.get("model", ""),
                    "status": "missing_in_model_registry",
                }
            )
            continue
        source = match.iloc[0]
        source_model = Path(source["model_path"])
        source_metadata = Path(source["metadata_path"])
        stem = safe_name(f"{row['task']}_{row['target']}_{row['model']}")
        copied_model = selected_dir / f"{stem}.joblib"
        copied_metadata = selected_dir / f"{stem}_metadata.json"
        status = "copied"
        if source_model.exists():
            shutil.copy2(source_model, copied_model)
        else:
            status = "missing_source_model"
        if source_metadata.exists():
            shutil.copy2(source_metadata, copied_metadata)
        else:
            status = "missing_source_metadata" if status == "copied" else status + "; missing_source_metadata"
        rows.append(
            {
                "task": row["task"],
                "target": row["target"],
                "model": row["model"],
                "model_family": row["model_family"],
                "composite_score": row.get("composite_score", np.nan),
                "source_model_path": str(source_model),
                "source_metadata_path": str(source_metadata),
                "selected_model_path": str(copied_model),
                "selected_metadata_path": str(copied_metadata),
                "status": status,
            }
        )
    pd.DataFrame(rows, columns=manifest_columns).to_csv(manifest_path, index=False)
    return manifest_path


if __name__ == "__main__":
    build_model_ranking()
