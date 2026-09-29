from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io_utils import read_yaml, sha256_file
from src.utils.logging_utils import get_logger
from src.utils.paths import config_dir, docs_dir, outputs_dir, project_root


LOGGER = get_logger("inspect_data")


VARIABLE_KEYWORDS = {
    "geometry": ["length", "width", "thickness", "height", "a0", "crack", "C_", "L_", "B_", "h_", "裂纹", "厚度", "高度"],
    "lever_arm_length": ["lever", "C_", "c(mm)", "探头到跨中", "C_Length"],
    "material_property": ["density", "moisture", "E1", "E2", "E3", "G12", "G13", "密度", "含水率"],
    "experimental_response": ["P", "load", "Load", "位移", "Delta", "Δ", "柔度", "area", "stiffness"],
    "fracture_parameter": ["Gi", "Gii", "GI", "GII", "Gc", "Gic", "Giic", "GII/G", "GI/GII"],
    "target_variable": ["Pmax", "GI_raw", "GII_raw", "Gc_raw", "Gi", "Gii"],
    "leakage_risk": ["Gii/G", "Gi/Gic", "Gii/Giic", "Gic", "Giic", "Gc", "ratio", "fraction"],
}


def classify_variable(name: str) -> str:
    hits: list[str] = []
    lower = str(name).lower()
    for category, keywords in VARIABLE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in lower or keyword in str(name):
                hits.append(category)
                break
    return "; ".join(hits) if hits else "needs_manual_review"


def summarize_series(s: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(s, errors="coerce")
    is_numeric = numeric.notna().sum() > 0 and numeric.notna().sum() >= max(1, int(0.5 * s.notna().sum()))
    row: dict[str, Any] = {
        "column_name": str(s.name),
        "dtype": str(s.dtype),
        "missing_count": int(s.isna().sum()),
        "unique_count": int(s.nunique(dropna=True)),
        "variable_type_guess": classify_variable(str(s.name)),
    }
    if is_numeric:
        row.update(
            {
                "min": float(numeric.min(skipna=True)),
                "max": float(numeric.max(skipna=True)),
                "mean": float(numeric.mean(skipna=True)),
                "std": float(numeric.std(skipna=True)),
            }
        )
    else:
        row.update({"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan})
    return row


def inspect_table(path: Path, sheet_name: str | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported table type for inspection: {path}")
    info = {
        "file": str(path),
        "sheet_name": sheet_name or "",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "sha256": sha256_file(path),
    }
    rows = []
    for col in df.columns:
        record = summarize_series(df[col])
        record.update(info)
        rows.append(record)
    return info, pd.DataFrame(rows)


def workbook_sheets(path: Path) -> list[str]:
    try:
        return pd.ExcelFile(path).sheet_names
    except Exception as exc:
        LOGGER.warning("Could not read workbook sheets for %s: %s", path, exc)
        return []


def run_inspection(max_extra_workbooks: int | None = 80) -> dict[str, Path]:
    cfg = read_yaml(config_dir() / "config.yaml")
    raw_root = Path(cfg["paths"]["raw_root"])
    docs_dir().mkdir(parents=True, exist_ok=True)
    report_dir = outputs_dir() / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    file_inventory_rows = []
    variable_frames = []
    processed_at = datetime.now().isoformat(timespec="seconds")

    selected = [
        Path(cfg["paths"]["main_workbook"]),
        Path(cfg["paths"]["corrected_results_workbook"]),
        Path(cfg["paths"]["correction_summary_workbook"]),
        Path(cfg["paths"]["physical_properties_workbook"]),
    ]
    all_tables = [p for p in raw_root.rglob("*") if p.is_file() and p.suffix.lower() in {".xlsx", ".xls", ".csv"}]
    scan_order = []
    for p in selected + all_tables:
        if p.exists() and p not in scan_order:
            scan_order.append(p)
    if max_extra_workbooks is not None:
        scan_order = selected + [p for p in scan_order if p not in selected][:max_extra_workbooks]

    for path in scan_order:
        rel = path.relative_to(raw_root) if path.is_relative_to(raw_root) else path
        base = {
            "file": str(path),
            "relative_to_raw_root": str(rel),
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "processed_at": processed_at,
        }
        if path.suffix.lower() in {".xlsx", ".xls"}:
            sheets = workbook_sheets(path)
            base["sheets"] = "; ".join(sheets)
            for sheet in sheets:
                try:
                    info, variables = inspect_table(path, sheet)
                    file_inventory_rows.append({**base, **info})
                    variable_frames.append(variables)
                except Exception as exc:
                    LOGGER.warning("Skipping sheet %s in %s: %s", sheet, path, exc)
                    file_inventory_rows.append({**base, "sheet_name": sheet, "error": str(exc)})
        elif path.suffix.lower() == ".csv":
            try:
                info, variables = inspect_table(path)
                file_inventory_rows.append({**base, **info})
                variable_frames.append(variables)
            except Exception as exc:
                LOGGER.warning("Skipping csv %s: %s", path, exc)
                file_inventory_rows.append({**base, "error": str(exc)})

    inventory = pd.DataFrame(file_inventory_rows)
    variable_dictionary = pd.concat(variable_frames, ignore_index=True) if variable_frames else pd.DataFrame()
    variable_dictionary_path = docs_dir() / "variable_dictionary.xlsx"
    mapping_template_path = docs_dir() / "variable_mapping_template.xlsx"
    provenance_path = docs_dir() / "data_provenance.md"
    inventory_path = report_dir / "raw_file_inventory.csv"

    inventory.to_csv(inventory_path, index=False)
    with pd.ExcelWriter(variable_dictionary_path) as writer:
        variable_dictionary.to_excel(writer, sheet_name="variable_dictionary", index=False)
        inventory.to_excel(writer, sheet_name="file_inventory", index=False)

    mapping_cols = [
        "source_file",
        "source_sheet",
        "original_column",
        "suggested_standard_name",
        "unit",
        "variable_role",
        "leakage_risk",
        "manual_confirmation",
        "notes",
    ]
    mapping_rows = []
    if not variable_dictionary.empty:
        for _, row in variable_dictionary.iterrows():
            if "needs_manual_review" in str(row["variable_type_guess"]) or "leakage_risk" in str(row["variable_type_guess"]):
                mapping_rows.append(
                    {
                        "source_file": row["file"],
                        "source_sheet": row["sheet_name"],
                        "original_column": row["column_name"],
                        "suggested_standard_name": "",
                        "unit": "",
                        "variable_role": row["variable_type_guess"],
                        "leakage_risk": "review_required" if "leakage_risk" in str(row["variable_type_guess"]) else "",
                        "manual_confirmation": "",
                        "notes": "",
                    }
                )
    pd.DataFrame(mapping_rows, columns=mapping_cols).to_excel(mapping_template_path, index=False)

    lines = [
        "# Data Provenance",
        "",
        f"- Project root: `{project_root()}`",
        f"- Raw data root: `{raw_root}`",
        f"- Inspection time: `{processed_at}`",
        f"- Script: `src/data/inspect_data.py`",
        f"- File inventory: `{inventory_path}`",
        f"- Variable dictionary: `{variable_dictionary_path}`",
        f"- Variable mapping template: `{mapping_template_path}`",
        "",
        "## Primary Data Sources",
    ]
    for key, value in cfg["paths"].items():
        if key.endswith("_dir") or key.endswith("_workbook") or key == "raw_root":
            p = Path(value)
            status = "found" if p.exists() else "missing"
            lines.append(f"- `{key}`: `{p}` ({status})")
    lines += [
        "",
        "## Processing Rule",
        "",
        "Raw files are treated as read-only. All cleaned data, model inputs, figures, tables and reports are generated under this project directory.",
    ]
    provenance_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Inspection complete: %s columns profiled", len(variable_dictionary))
    return {
        "inventory": inventory_path,
        "variable_dictionary": variable_dictionary_path,
        "mapping_template": mapping_template_path,
        "provenance": provenance_path,
    }


if __name__ == "__main__":
    run_inspection()
