from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io_utils import read_yaml, save_table
from src.utils.logging_utils import get_logger
from src.utils.paths import config_dir, data_dir


LOGGER = get_logger("clean_data")


RAW_TO_STANDARD = {
    "实验编号": "specimen_id",
    "组别": "group",
    "试样编号": "sample_no",
    "源文件": "source_file",
    "源sheet行号": "source_sheet_row",
    "C_探头到跨中": "lever_arm_length",
    "L_跨中到端部": "span_length",
    "E1(MPa)": "E1_MPa",
    "E2(MPa)": "E2_MPa",
    "E3(MPa)": "E3_MPa",
    "G12(MPa)": "G12_MPa",
    "G13(MPa)": "G13_MPa",
    "h_高度": "specimen_height",
    "B_厚度": "specimen_width",
    "a0_初始裂纹长度": "initial_crack_length",
    "P(原始值)": "load_raw",
    "P（修正值）": "load_corrected",
    "PI": "load_mode_I_component",
    "PII": "load_mode_II_component",
    "端部位移Δa": "end_displacement",
    "跨中位移Δb": "mid_displacement",
    "Δ1": "delta1",
    "Δ2": "delta2",
    "柔度系数CI": "compliance_CI_point",
    "柔度系数CII": "compliance_CII_point",
    "C2corr": "C2corr",
    "aeq1": "aeq1",
    "aeqii": "aeqii",
    "Gi": "GI_point",
    "Gii": "GII_point",
    "Gii/G": "GII_fraction_nominal",
    "Gi/Gic": "GI_over_GIC_nominal",
    "Gii/Giic": "GII_over_GIIC_nominal",
    "Gic": "GIC",
    "Giic": "GIIC",
}


def numeric_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    exclude = exclude or set()
    out = df.copy()
    for col in out.columns:
        if col not in exclude:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().sum() >= max(1, int(0.4 * out[col].notna().sum())):
                out[col] = converted
    return out


def add_standard_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=RAW_TO_STANDARD).copy()
    out = numeric_columns(out, exclude={"specimen_id", "group", "source_file"})
    out["group"] = out["group"].astype(str)
    out["direction"] = out["group"].str[0].map({"R": "R", "T": "T"})
    out["lever_arm_label"] = out["group"].str[-1].map({"A": "A", "B": "B", "C": "C"})
    out["direction_R"] = (out["direction"] == "R").astype(int)
    out["direction_T"] = (out["direction"] == "T").astype(int)
    out["specimen_thickness"] = out["specimen_height"]
    # Material properties belong in the user's input data, never in public code.
    for column in ("density", "moisture_content", "loading_rate"):
        if column not in out:
            out[column] = np.nan
    if "load_corrected" in out.columns and "load_raw" in out.columns:
        out["load_for_analysis"] = out["load_corrected"].fillna(out["load_raw"])
    elif "load_corrected" in out.columns:
        out["load_for_analysis"] = out["load_corrected"]
    elif "load_raw" in out.columns:
        out["load_for_analysis"] = out["load_raw"]
    else:
        out["load_for_analysis"] = np.nan
    out["row_in_specimen"] = out.groupby("specimen_id").cumcount() + 1
    out["row_count_specimen"] = out.groupby("specimen_id")["row_in_specimen"].transform("max")
    out["row_fraction_specimen"] = out["row_in_specimen"] / out["row_count_specimen"].replace(0, np.nan)
    return out


def load_main_workbook() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = read_yaml(config_dir() / "config.yaml")
    main = Path(cfg["paths"]["main_workbook"])
    point = pd.read_excel(main, sheet_name=cfg["data"]["main_sheet"])
    params = pd.read_excel(main, sheet_name=cfg["data"]["parameter_sheet"])
    return add_standard_fields(point), add_standard_fields(params)


def build_clean_datasets() -> dict[str, Path]:
    processed = data_dir() / "processed"
    interim = data_dir() / "interim"
    processed.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)

    point, params = load_main_workbook()
    positive = point.dropna(subset=["GI_point", "GII_point"]).copy()
    positive = positive[(positive["GI_point"] > 0) & (positive["GII_point"] > 0)].copy()

    paths = {
        "point_all": interim / "point_data_all_rows.csv",
        "point_positive": processed / "point_data_positive_GI_GII.csv",
        "specimen_constants": processed / "specimen_constants.csv",
    }
    save_table(point, paths["point_all"])
    save_table(positive, paths["point_positive"])
    save_table(params.drop_duplicates("specimen_id"), paths["specimen_constants"])
    LOGGER.info("Clean datasets written: all=%s, positive=%s", point.shape, positive.shape)
    return paths


if __name__ == "__main__":
    build_clean_datasets()
