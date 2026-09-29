from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.compute_mmb_parameters import compute_specimen_parameters
from src.features.mechanics_features import add_mechanics_features
from src.utils.io_utils import read_yaml, save_table
from src.utils.logging_utils import get_logger
from src.utils.paths import config_dir, data_dir, outputs_dir


LOGGER = get_logger("build_features")


def load_modeling_dataset(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        path = data_dir() / "processed" / "modeling_dataset_specimen_level.csv"
    if not path.exists() and path == data_dir() / "processed" / "modeling_dataset_specimen_level.csv":
        compute_specimen_parameters()
    df = pd.read_csv(path)
    return add_mechanics_features(df)


def get_task_features(feature_cfg: dict, task_name: str, target: str, available_columns: list[str]) -> tuple[list[str], list[str]]:
    task = feature_cfg["tasks"][task_name]
    raw_features: list[str] = []
    for set_name in task["feature_sets"]:
        raw_features.extend(feature_cfg["feature_sets"][set_name]["features"])
    raw_features = list(dict.fromkeys(raw_features))
    derived = set(feature_cfg.get("target_derived_variables", {}).get(target, []))
    always_exclude = set(feature_cfg.get("always_exclude_from_inputs", []))
    excluded = []
    features = []
    for feature in raw_features:
        reason = None
        if feature == target:
            reason = "target_itself"
        elif feature in derived:
            reason = "target_derived"
        elif feature in always_exclude:
            reason = "always_excluded"
        elif feature not in available_columns:
            reason = "missing_column"
        if reason:
            excluded.append(f"{feature}:{reason}")
        else:
            features.append(feature)
    return features, excluded


def build_feature_manifest() -> Path:
    df = load_modeling_dataset()
    feature_cfg = read_yaml(config_dir() / "feature_sets.yaml")
    rows = []
    for task_name, task in feature_cfg["tasks"].items():
        for target in task["targets"]:
            features, excluded = get_task_features(feature_cfg, task_name, target, list(df.columns))
            rows.append(
                {
                    "task": task_name,
                    "target": target,
                    "n_features": len(features),
                    "features": ", ".join(features),
                    "excluded_features_due_to_leakage_or_missing": ", ".join(excluded),
                }
            )
    manifest = pd.DataFrame(rows)
    out_dir = outputs_dir() / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "feature_manifest.csv"
    save_table(manifest, path)
    save_table(df, data_dir() / "processed" / "modeling_dataset_with_features.csv")
    LOGGER.info("Feature manifest written: %s", path)
    return path


if __name__ == "__main__":
    build_feature_manifest()
