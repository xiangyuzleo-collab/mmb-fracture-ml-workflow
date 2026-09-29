from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import outputs_dir


METRIC_COLUMNS = ["R2", "MAE", "RMSE", "MAPE", "MaxAE", "MaxRE", "MeanRE", "MedianRE"]


def _summarize_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    present_metrics = [col for col in METRIC_COLUMNS if col in df.columns]
    grouped = df.groupby(group_cols)[present_metrics].agg(["mean", "std", "min", "max"]).reset_index()
    grouped.columns = ["_".join([str(x) for x in col if x]) if isinstance(col, tuple) else str(col) for col in grouped.columns]
    if "fold" in df.columns:
        folds = df.groupby(group_cols)["fold"].nunique().reset_index(name="n_folds")
        grouped = grouped.merge(folds, on=group_cols, how="left")
    return grouped


def export_validation_tables() -> dict[str, Path]:
    out_dir = outputs_dir() / "tables" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    rrs_path = outputs_dir() / "metrics" / "repeated_random_split" / "repeated_random_split_metrics.csv"
    if rrs_path.exists():
        rrs = pd.read_csv(rrs_path)
        rrs_summary = _summarize_metrics(rrs, ["task", "target", "model", "model_family"])
        out = out_dir / "repeated_random_split_summary.xlsx"
        with pd.ExcelWriter(out) as writer:
            rrs_summary.to_excel(writer, sheet_name="summary_mean_std", index=False)
            rrs.to_excel(writer, sheet_name="all_folds", index=False)
        paths["repeated_random_split_summary"] = out

    loo_path = outputs_dir() / "metrics" / "leave_one_lever_arm_out" / "leave_one_lever_arm_out_metrics.csv"
    if loo_path.exists():
        loo = pd.read_csv(loo_path)
        loo_summary = _summarize_metrics(
            loo,
            ["task", "target", "model", "model_family", "lever_arm_length_left_out", "generalization_type"],
        )
        out = out_dir / "leave_one_lever_arm_out_summary.xlsx"
        with pd.ExcelWriter(out) as writer:
            loo_summary.to_excel(writer, sheet_name="summary_by_left_out_c", index=False)
            loo.to_excel(writer, sheet_name="all_cases", index=False)
        paths["leave_one_lever_arm_out_summary"] = out

    ranking_path = outputs_dir() / "tables" / "model_comparison" / "final_model_ranking.xlsx"
    if ranking_path.exists():
        ranking = pd.read_excel(ranking_path, sheet_name="final_model_ranking")
        best = ranking[ranking["rank_within_task_target"] == 1].copy() if "rank_within_task_target" in ranking else ranking
        out = out_dir / "best_model_by_task_target.xlsx"
        with pd.ExcelWriter(out) as writer:
            best.to_excel(writer, sheet_name="best_model_by_task_target", index=False)
            ranking.to_excel(writer, sheet_name="full_ranking", index=False)
        paths["best_model_by_task_target"] = out

    return paths


if __name__ == "__main__":
    export_validation_tables()
