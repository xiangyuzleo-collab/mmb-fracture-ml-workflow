from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_data import build_clean_datasets
from src.utils.io_utils import read_yaml, save_table
from src.utils.logging_utils import get_logger
from src.utils.paths import config_dir, data_dir, docs_dir, outputs_dir


LOGGER = get_logger("compute_mmb_parameters")


def _normalize_source_file(value: object) -> str:
    """Normalize duplicate-download suffixes without changing specimen identity."""
    name = Path(str(value).strip()).name.lower()
    return re.sub(r"\s*\(\d+\)(?=\.xlsx?$)", "", name)


def _safe_polyfit_stiffness(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    data = data[(data["x"] >= 0) & (data["y"] >= 0)].sort_values("x")
    if len(data) < 5 or data["x"].nunique() < 2:
        return float("nan")
    ymax = data["y"].max()
    window = data[(data["y"] >= 0.05 * ymax) & (data["y"] <= 0.35 * ymax)]
    if len(window) < 5:
        window = data.head(min(12, len(data)))
    try:
        slope, _ = np.polyfit(window["x"], window["y"], 1)
        return float(slope)
    except Exception:
        return float("nan")


def _area(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    data = data[(data["x"] >= 0) & (data["y"] >= 0)].sort_values("x")
    if len(data) < 2:
        return float("nan")
    return float(np.trapz(data["y"], data["x"]))


def _correction_summary() -> pd.DataFrame:
    cfg = read_yaml(config_dir() / "config.yaml")
    path = Path(cfg["paths"]["correction_summary_workbook"])
    if not path.exists():
        return pd.DataFrame(columns=["source_file", "GI_corrected_summary", "GII_corrected_summary"])
    df = pd.read_excel(path)
    df = df.rename(columns={"文件名": "source_file", "Gi": "GI_corrected_summary", "Gii": "GII_corrected_summary"})
    df["source_file_key"] = df["source_file"].map(_normalize_source_file)
    if df["source_file_key"].duplicated().any():
        duplicates = df.loc[df["source_file_key"].duplicated(keep=False), "source_file"].tolist()
        raise ValueError(f"Duplicate correction-summary specimen keys after normalization: {duplicates}")
    return df[["source_file", "source_file_key", "GI_corrected_summary", "GII_corrected_summary"]]


def compute_specimen_parameters() -> dict[str, Path]:
    processed = data_dir() / "processed"
    if not (processed / "point_data_positive_GI_GII.csv").exists():
        build_clean_datasets()
    point_path = processed / "point_data_all_rows.csv"
    if not point_path.exists():
        point_path = data_dir() / "interim" / "point_data_all_rows.csv"
    point = pd.read_csv(point_path)
    constants = pd.read_csv(processed / "specimen_constants.csv")
    corrections = _correction_summary()

    rows = []
    for specimen_id, sub in point.groupby("specimen_id", dropna=False):
        sub = sub.copy()
        load = pd.to_numeric(sub["load_for_analysis"], errors="coerce")
        pmax_idx = load.idxmax() if load.notna().any() else sub.index[0]
        peak = sub.loc[pmax_idx]
        gi_point = pd.to_numeric(sub["GI_point"], errors="coerce")
        gii_point = pd.to_numeric(sub["GII_point"], errors="coerce")
        gc_point = gi_point + gii_point
        row = {
            "specimen_id": specimen_id,
            "group": peak.get("group"),
            "direction": peak.get("direction"),
            "lever_arm_label": peak.get("lever_arm_label"),
            "lever_arm_length": peak.get("lever_arm_length"),
            "sample_no": peak.get("sample_no"),
            "source_file": peak.get("source_file"),
            "Pmax": float(load.max(skipna=True)),
            "peak_end_displacement": peak.get("end_displacement"),
            "peak_mid_displacement": peak.get("mid_displacement"),
            "peak_delta1": peak.get("delta1"),
            "peak_delta2": peak.get("delta2"),
            "peak_displacement": peak.get("end_displacement"),
            "initial_stiffness": _safe_polyfit_stiffness(sub["end_displacement"], load),
            "area_load_end_displacement": _area(sub["end_displacement"], load),
            "area_load_mid_displacement": _area(sub["mid_displacement"], load),
            "GI_point_max": float(gi_point.max(skipna=True)),
            "GII_point_max": float(gii_point.max(skipna=True)),
            "Gc_point_max_simultaneous": float(gc_point.max(skipna=True)),
            "GI_point_mean_positive": float(gi_point[gi_point > 0].mean(skipna=True)),
            "GII_point_mean_positive": float(gii_point[gii_point > 0].mean(skipna=True)),
            "positive_GI_GII_point_count": int(((gi_point > 0) & (gii_point > 0)).sum()),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["compliance"] = 1.0 / summary["initial_stiffness"].replace(0, np.nan)
    summary["source_file_key"] = summary["source_file"].map(_normalize_source_file)
    summary = summary.merge(
        corrections.drop(columns=["source_file"]),
        on="source_file_key",
        how="left",
        validate="one_to_one",
    )
    missing_corrections = summary[
        summary[["GI_corrected_summary", "GII_corrected_summary"]].isna().any(axis=1)
    ][["specimen_id", "source_file"]]
    if not missing_corrections.empty:
        raise ValueError(
            "Correction-summary matching is incomplete; no fallback to pointwise maxima is allowed:\n"
            + missing_corrections.to_string(index=False)
        )
    const_cols = [
        "specimen_id",
        "span_length",
        "E1_MPa",
        "E2_MPa",
        "E3_MPa",
        "G12_MPa",
        "G13_MPa",
        "specimen_height",
        "specimen_width",
        "specimen_thickness",
        "initial_crack_length",
        "density",
        "moisture_content",
        "loading_rate",
        "GIC",
        "GIIC",
        "GII_fraction_nominal",
        "GI_over_GIC_nominal",
        "GII_over_GIIC_nominal",
    ]
    const_present = [c for c in const_cols if c in constants.columns]
    summary = summary.merge(constants[const_present].drop_duplicates("specimen_id"), on="specimen_id", how="left")

    # The publication targets use one consistent critical/corrected definition for all specimens.
    # Pointwise peak responses are retained in separate audit columns and are never mixed into GI_raw/GII_raw.
    summary["GI_raw"] = summary["GI_corrected_summary"]
    summary["GII_raw"] = summary["GII_corrected_summary"]
    summary["Gc_raw"] = summary["GI_raw"] + summary["GII_raw"]
    summary["GI_calculated"] = summary["GI_point_max"]
    summary["GII_calculated"] = summary["GII_point_max"]
    summary["Gc_calculated_sum_of_component_max"] = summary["GI_calculated"] + summary["GII_calculated"]
    summary["Gc_calculated"] = summary["Gc_point_max_simultaneous"]
    summary["GI_GII_ratio"] = summary["GI_raw"] / summary["GII_raw"].replace(0, np.nan)
    summary["GII_fraction"] = summary["GII_raw"] / summary["Gc_raw"].replace(0, np.nan)

    # Mechanics-informed nondimensional features.
    summary["lever_arm_over_thickness"] = summary["lever_arm_length"] / summary["specimen_thickness"].replace(0, np.nan)
    summary["lever_arm_over_span"] = summary["lever_arm_length"] / summary["span_length"].replace(0, np.nan)
    summary["initial_crack_over_length"] = summary["initial_crack_length"] / summary["span_length"].replace(0, np.nan)
    summary["initial_crack_over_thickness"] = summary["initial_crack_length"] / summary["specimen_thickness"].replace(0, np.nan)
    summary["width_over_thickness"] = summary["specimen_width"] / summary["specimen_thickness"].replace(0, np.nan)
    summary["height_over_thickness"] = summary["specimen_height"] / summary["specimen_thickness"].replace(0, np.nan)
    summary["direction_R"] = (summary["direction"] == "R").astype(int)
    summary["direction_T"] = (summary["direction"] == "T").astype(int)

    checks = summary.copy()
    checks["GI_positive"] = checks["GI_raw"] > 0
    checks["GII_positive"] = checks["GII_raw"] > 0
    checks["Gc_equals_GI_plus_GII_error"] = (checks["Gc_raw"] - checks["GI_raw"] - checks["GII_raw"]).abs()
    checks["ratio_outlier_flag"] = ~checks["GI_GII_ratio"].between(0, 20)

    tables = outputs_dir() / "tables" / "experimental"
    tables.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    all_path = tables / "mmb_parameters_all_samples.xlsx"
    by_lever_path = tables / "experimental_summary_by_lever_arm.xlsx"
    processed_path = processed / "modeling_dataset_specimen_level.csv"
    checks_path = outputs_dir() / "reports" / "mmb_parameter_reasonableness_checks.csv"

    group_cols = ["direction", "lever_arm_length"]
    numeric_cols = ["Pmax", "peak_displacement", "initial_stiffness", "compliance", "GI_raw", "GII_raw", "Gc_raw", "GI_GII_ratio", "GII_fraction"]
    grouped = summary.groupby(group_cols)[numeric_cols].agg(["count", "mean", "std", "min", "max"])
    grouped.columns = ["_".join([a, b]) for a, b in grouped.columns]
    grouped = grouped.reset_index()
    for col in numeric_cols:
        if f"{col}_mean" in grouped and f"{col}_std" in grouped:
            grouped[f"{col}_cv"] = grouped[f"{col}_std"] / grouped[f"{col}_mean"].replace(0, np.nan)

    save_table(summary, all_path)
    save_table(grouped, by_lever_path)
    save_table(summary, processed_path)
    save_table(checks, checks_path)
    write_modeling_assumptions()
    LOGGER.info("MMB specimen parameters written: %s", summary.shape)
    return {
        "all_samples": all_path,
        "summary_by_lever": by_lever_path,
        "modeling_dataset": processed_path,
        "reasonableness_checks": checks_path,
    }


def write_modeling_assumptions() -> None:
    docs_dir().mkdir(parents=True, exist_ok=True)
    path = docs_dir() / "modeling_assumptions.md"
    path.write_text(
        "\n".join(
            [
                "# Modeling Assumptions",
                "",
                "1. Raw data files under the original MMB directory are treated as read-only.",
                "2. `GI_raw` and `GII_raw` are taken from the configured correction summary after normalizing duplicate-download filename suffixes such as `(1)`. Missing matches raise an error; pointwise maxima are never used as a silent fallback.",
                "3. `GI_calculated` and `GII_calculated` are not independent re-derivations of the ASTM/CBT/MMB formula. They are separate pointwise component maxima retained for audit. `Gc_calculated` is the maximum simultaneous value of `GI_point + GII_point`; the sum of separate component maxima is retained as `Gc_calculated_sum_of_component_max`.",
                "4. A full independent MMB formula implementation requires final confirmation of the exact geometry definitions, compliance correction and Liu-style calculation path. Until confirmed, the script records the formula-dependent values as source-derived rather than overwriting the original values.",
                "5. Initial stiffness is estimated by a linear fit to the early positive load-displacement region, approximately 5%-35% of specimen peak load. The estimate is used as a response descriptor rather than a standard-code material constant.",
                "6. Load-displacement area is computed by trapezoidal integration after sorting by the corresponding displacement channel and removing negative load/displacement points.",
                "7. Density and moisture content must come from the supplied input data; missing values remain missing.",
                "8. Any intermediate lever-arm recommendation is an interpolation hypothesis, not an experimentally verified optimum.",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    compute_specimen_parameters()
