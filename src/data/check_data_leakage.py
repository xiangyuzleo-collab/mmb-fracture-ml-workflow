from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io_utils import dataframe_to_markdown, read_yaml
from src.utils.logging_utils import get_logger
from src.utils.paths import config_dir, docs_dir


LOGGER = get_logger("check_data_leakage")


def expand_features(feature_cfg: dict, task_name: str) -> list[str]:
    task = feature_cfg["tasks"][task_name]
    features: list[str] = []
    for set_name in task["feature_sets"]:
        features.extend(feature_cfg["feature_sets"][set_name]["features"])
    return list(dict.fromkeys(features))


def leakage_for_task(feature_cfg: dict, task_name: str, target: str) -> dict:
    features = set(expand_features(feature_cfg, task_name))
    direct = target in features
    derived = sorted(features.intersection(set(feature_cfg.get("target_derived_variables", {}).get(target, []))))
    always_excluded = sorted(features.intersection(set(feature_cfg.get("always_exclude_from_inputs", []))))
    safe_features = [f for f in expand_features(feature_cfg, task_name) if f != target and f not in derived and f not in always_excluded]
    return {
        "task": task_name,
        "target": target,
        "direct_target_in_inputs": direct,
        "target_derived_inputs": ", ".join(derived),
        "always_excluded_present": ", ".join(always_excluded),
        "safe_feature_count_after_filter": len(safe_features),
        "safe_features_after_filter": ", ".join(safe_features),
        "status": "FAIL" if direct or derived or always_excluded else "PASS",
    }


def run_leakage_check() -> Path:
    feature_cfg = read_yaml(config_dir() / "feature_sets.yaml")
    rows = []
    for task_name, task in feature_cfg["tasks"].items():
        for target in task["targets"]:
            rows.append(leakage_for_task(feature_cfg, task_name, target))
    report = pd.DataFrame(rows)
    docs_dir().mkdir(parents=True, exist_ok=True)
    xlsx_path = docs_dir() / "leakage_check_details.xlsx"
    md_path = docs_dir() / "leakage_check_report.md"
    report.to_excel(xlsx_path, index=False)

    lines = [
        "# Leakage Check Report",
        "",
        "This report checks configured feature sets before model fitting. The implementation also fits scalers inside sklearn `Pipeline` objects after each train/test split.",
        "",
        f"- Total task-target checks: {len(report)}",
        f"- Failed configuration checks before automatic filtering: {(report['status'] == 'FAIL').sum()}",
        f"- Detailed workbook: `{xlsx_path}`",
        "",
        "## Rules",
        "",
        "1. The target variable itself must not be in the input feature list.",
        "2. Variables derived from the target, such as `Gc_raw`, `GI_GII_ratio`, and `GII_fraction`, are excluded when they would leak the target.",
        "3. Specimen identifiers and source-file fields are never used as predictive inputs.",
        "4. Standardization and preprocessing are performed inside sklearn Pipelines and are fitted only on training folds.",
        "5. Repeated random split works at the specimen level. The project does not split pointwise rows from the same specimen across train/test for the main small-sample tasks.",
        "6. Leave-one-lever-arm validation holds out all specimens at a given lever-arm length.",
        "",
        "## Configuration Findings",
        "",
        dataframe_to_markdown(report[["task", "target", "status", "target_derived_inputs", "safe_feature_count_after_filter"]]),
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Leakage report written: %s", md_path)
    return md_path


if __name__ == "__main__":
    run_leakage_check()
