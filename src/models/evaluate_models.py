from __future__ import annotations

import argparse
import copy
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import get_task_features, load_modeling_dataset
from src.features.feature_selection import remove_constant_features
from src.models.model_registry import ModelSpec, build_model_specs
from src.models.save_load_models import save_model
from src.utils.io_utils import read_yaml, safe_name, sha256_dataframe, write_json
from src.utils.logging_utils import get_logger
from src.utils.metrics import metric_rows
from src.utils.paths import config_dir, outputs_dir
from src.utils.random_seed import set_random_seed
from src.validation.cross_validation_utils import require_specimen_level_rows, save_split, summarize_splits
from src.validation.leave_one_lever_arm_out import leave_one_lever_arm_splits
from src.validation.repeated_random_split import repeated_random_splits


LOGGER = get_logger("evaluate_models")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _strategy_dirs(strategy: str, output_root: Path | None = None) -> dict[str, Path]:
    root = output_root if output_root is not None else outputs_dir()
    return {
        "metrics": root / "metrics" / strategy,
        "predictions": root / "predictions" / strategy,
        "splits": root / "metrics" / strategy / "splits",
    }


def _model_path(family: str, model_name: str, stem: str, output_root: Path | None = None) -> Path:
    root = output_root if output_root is not None else outputs_dir()
    return root / "models" / family / model_name / f"{stem}.joblib"


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + "_metadata.json")


def _split_records(df: pd.DataFrame, strategy: str, n_repeats: int | None = None) -> list[dict]:
    cfg = read_yaml(config_dir() / "config.yaml")
    seed = int(cfg["project"]["random_seed"])
    if strategy == "repeated_random_split":
        vcfg = cfg["validation"]["repeated_random_split"]
        strata = (
            df["direction"].astype(str)
            + "_c"
            + pd.to_numeric(df["lever_arm_length"], errors="coerce").astype("Int64").astype(str)
        )
        return list(
            repeated_random_splits(
                df.index,
                n_repeats=n_repeats or int(vcfg["n_repeats"]),
                test_size=float(vcfg["test_size"]),
                random_seed=seed,
                strata=strata,
            )
        )
    if strategy == "leave_one_lever_arm_out":
        return list(leave_one_lever_arm_splits(df))
    raise ValueError(f"Unknown validation strategy: {strategy}")


def _metadata(
    spec: ModelSpec,
    task_name: str,
    target: str,
    features: list[str],
    excluded: list[str],
    data_hash: str,
    split: dict,
    strategy: str,
    metrics: dict,
    script_used: str,
    data_file: str = "data/processed/modeling_dataset_with_features.csv",
) -> dict:
    return {
        "model_name": spec.name,
        "model_family": spec.family,
        "task_name": task_name,
        "target_variable": target,
        "feature_set": task_name,
        "input_features": features,
        "excluded_features_due_to_leakage": excluded,
        "data_file": data_file,
        "data_hash": data_hash,
        "train_indices": split.get("train_indices", []),
        "test_indices": split.get("test_indices", []),
        "validation_strategy": strategy,
        "random_seed": int(read_yaml(config_dir() / "config.yaml")["project"]["random_seed"]),
        "hyperparameters": spec.hyperparameters,
        "metrics": metrics,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script_used": script_used,
    }


def _fit_predict(spec: ModelSpec, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    estimator = copy.deepcopy(spec.estimator)
    estimator.fit(X_train, y_train)
    return estimator, np.asarray(estimator.predict(X_test), dtype=float)


def _validate_selection(
    model_cfg: dict,
    feature_cfg: dict,
    only_models: list[str] | None,
    only_tasks: list[str] | None,
    only_targets: list[str] | None,
) -> None:
    available = {
        "models": set(model_cfg["models"]),
        "tasks": set(feature_cfg["tasks"]),
        "targets": {target for task in feature_cfg["tasks"].values() for target in task["targets"]},
    }
    for label, selected in (("models", only_models), ("tasks", only_tasks), ("targets", only_targets)):
        if selected is not None and not selected:
            raise ValueError(f"No {label} selected")
        unknown = set(selected or []) - available[label]
        if unknown:
            raise ValueError(f"Unknown {label}: {', '.join(sorted(unknown))}")


def run_validation(
    strategy: str,
    n_repeats: int | None = None,
    only_models: list[str] | None = None,
    only_tasks: list[str] | None = None,
    only_targets: list[str] | None = None,
    save_fold_models: bool | None = None,
    dataset_path: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Path]:
    cfg = read_yaml(config_dir() / "config.yaml")
    feature_cfg = read_yaml(config_dir() / "feature_sets.yaml")
    model_cfg = read_yaml(config_dir() / "model_config.yaml")
    _validate_selection(model_cfg, feature_cfg, only_models, only_tasks, only_targets)
    seed = int(cfg["project"]["random_seed"])
    set_random_seed(seed)

    df = load_modeling_dataset(dataset_path).reset_index(drop=True)
    require_specimen_level_rows(df)
    data_hash = sha256_dataframe(df)
    specs = build_model_specs(model_cfg, random_seed=seed, only=only_models)
    unavailable = [s for s in specs if not s.optional_dependency_available]
    specs = [s for s in specs if s.optional_dependency_available and s.estimator is not None]
    if unavailable:
        LOGGER.warning("Skipping optional unavailable models: %s", [s.name for s in unavailable])
    if not specs:
        raise ValueError("No available models selected for validation")

    dirs = _strategy_dirs(strategy, output_root)
    split_records = _split_records(df, strategy, n_repeats=n_repeats)

    metrics_rows = []
    pred_rows = []
    if save_fold_models is None:
        save_fold_models = bool(model_cfg.get("save_fold_models", {}).get(strategy, False))

    for task_name, task in feature_cfg["tasks"].items():
        if only_tasks and task_name not in only_tasks:
            continue
        for target in task["targets"]:
            if only_targets and target not in only_targets:
                continue
            if target not in df.columns:
                LOGGER.warning("Skipping missing target %s", target)
                continue
            features, excluded = get_task_features(feature_cfg, task_name, target, list(df.columns))
            if not features:
                LOGGER.warning("Skipping %s/%s because no safe features remain", task_name, target)
                continue
            meta_cols = ["specimen_id", "group", "direction", "lever_arm_length"]
            selected_cols = list(dict.fromkeys(meta_cols + [target] + features))
            modeling = df[selected_cols].dropna(subset=[target])
            if len(modeling) < 8:
                LOGGER.warning("Skipping %s/%s because sample size is %s", task_name, target, len(modeling))
                continue
            for split in split_records:
                train_idx = [i for i in split["train_indices"] if i in modeling.index]
                test_idx = [i for i in split["test_indices"] if i in modeling.index]
                if len(train_idx) < 5 or len(test_idx) < 2:
                    continue
                X_train, y_train = modeling.loc[train_idx, features], modeling.loc[train_idx, target]
                X_test, y_test = modeling.loc[test_idx, features], modeling.loc[test_idx, target]
                fold_features, removed_constant = remove_constant_features(X_train, features)
                if not fold_features:
                    LOGGER.warning("Skipping %s/%s/%s because no training-fold features remain", task_name, target, split["fold"])
                    continue
                X_train, X_test = X_train[fold_features], X_test[fold_features]
                for spec in specs:
                    try:
                        estimator, y_pred = _fit_predict(spec, X_train, y_train, X_test)
                    except Exception as exc:
                        LOGGER.warning("Model failed: %s %s %s %s: %s", strategy, task_name, target, spec.name, exc)
                        continue
                    row = metric_rows(
                        y_test,
                        y_pred,
                        validation_strategy=strategy,
                        fold=split["fold"],
                        model=spec.name,
                        model_family=spec.family,
                        task=task_name,
                        target=target,
                        n_train=len(train_idx),
                        n_test=len(test_idx),
                        lever_arm_length_left_out=split.get("lever_arm_length_left_out", np.nan),
                        generalization_type=split.get("generalization_type", ""),
                    )
                    metrics_rows.append(row)
                    for idx, actual, pred in zip(test_idx, y_test.to_numpy(), y_pred):
                        meta = modeling.loc[idx, ["specimen_id", "group", "direction", "lever_arm_length"]].to_dict()
                        pred_rows.append(
                            {
                                **meta,
                                "row_index": int(idx),
                                "validation_strategy": strategy,
                                "fold": split["fold"],
                                "model": spec.name,
                                "model_family": spec.family,
                                "task": task_name,
                                "target": target,
                                "actual": float(actual),
                                "predicted": float(pred),
                                "residual": float(pred - actual),
                                "relative_error_percent": float(abs(pred - actual) / abs(actual) * 100.0)
                                if abs(actual) > 1e-12
                                else np.nan,
                            }
                        )
                    if save_fold_models:
                        stem = safe_name(f"{task_name}_{target}_{strategy}_{split['fold']}_{spec.name}")
                        model_path = _model_path(spec.family, spec.name, stem, output_root)
                        save_model(estimator, model_path)
                        write_json(
                            _metadata(
                                spec, task_name, target, fold_features,
                                excluded + removed_constant, data_hash, split,
                                strategy, row, __file__,
                                "examples/synthetic_specimens.csv" if dataset_path else "data/processed/modeling_dataset_with_features.csv",
                            ),
                            _metadata_path(model_path),
                        )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(pred_rows)
    if metrics.empty:
        raise RuntimeError("Validation produced no metrics; check target availability and split sizes")
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    split_summary = summarize_splits(split_records)
    split_summary.to_csv(dirs["splits"] / "split_summary.csv", index=False)
    for split in split_records:
        save_split(split, dirs["splits"] / f"{split['fold']}.json")
    metrics_path = dirs["metrics"] / f"{strategy}_metrics.csv"
    predictions_path = dirs["predictions"] / f"{strategy}_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    LOGGER.info("Validation complete: %s metrics rows, %s prediction rows", len(metrics), len(predictions))
    return {"metrics": metrics_path, "predictions": predictions_path, "split_summary": dirs["splits"] / "split_summary.csv"}


def refit_full_models(only_models: list[str] | None = None, only_tasks: list[str] | None = None, only_targets: list[str] | None = None) -> Path:
    cfg = read_yaml(config_dir() / "config.yaml")
    feature_cfg = read_yaml(config_dir() / "feature_sets.yaml")
    model_cfg = read_yaml(config_dir() / "model_config.yaml")
    seed = int(cfg["project"]["random_seed"])
    df = load_modeling_dataset().reset_index(drop=True)
    data_hash = sha256_dataframe(df)
    specs = [s for s in build_model_specs(model_cfg, random_seed=seed, only=only_models) if s.optional_dependency_available and s.estimator is not None]
    rows = []
    for task_name, task in feature_cfg["tasks"].items():
        if only_tasks and task_name not in only_tasks:
            continue
        for target in task["targets"]:
            if only_targets and target not in only_targets or target not in df.columns:
                continue
            features, excluded = get_task_features(feature_cfg, task_name, target, list(df.columns))
            features, removed = remove_constant_features(df, features)
            excluded = excluded + removed
            meta_cols = ["specimen_id", "group", "direction", "lever_arm_length"]
            selected_cols = list(dict.fromkeys(meta_cols + [target] + features))
            modeling = df[selected_cols].dropna(subset=[target])
            if len(modeling) < 8 or not features:
                continue
            X, y = modeling[features], modeling[target]
            split = {"fold": "full_data_refit", "train_indices": modeling.index.tolist(), "test_indices": []}
            for spec in specs:
                try:
                    estimator = copy.deepcopy(spec.estimator)
                    estimator.fit(X, y)
                    pred = np.asarray(estimator.predict(X), dtype=float)
                    train_metrics = metric_rows(y, pred)
                except Exception as exc:
                    LOGGER.warning("Full refit failed: %s/%s/%s: %s", task_name, target, spec.name, exc)
                    continue
                stem = safe_name(f"{task_name}_{target}_full_data_{spec.name}")
                model_path = _model_path(spec.family, spec.name, stem)
                save_model(estimator, model_path)
                metadata = _metadata(
                    spec,
                    task_name,
                    target,
                    features,
                    excluded,
                    data_hash,
                    split,
                    "full_data_refit_after_validation",
                    train_metrics,
                    __file__,
                )
                write_json(metadata, _metadata_path(model_path))
                rows.append(
                    {
                        "task": task_name,
                        "target": target,
                        "model": spec.name,
                        "model_family": spec.family,
                        "model_path": str(model_path),
                        "metadata_path": str(_metadata_path(model_path)),
                        "n_samples": len(modeling),
                        "n_features": len(features),
                        **train_metrics,
                    }
                )
    registry = pd.DataFrame(rows)
    path = outputs_dir() / "models" / "model_registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(path, index=False)
    LOGGER.info("Full model registry written: %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["repeated_random_split", "leave_one_lever_arm_out"], required=True)
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument("--refit-full", action="store_true")
    args = parser.parse_args()
    run_validation(args.strategy, args.n_repeats, args.models, args.tasks, args.targets, args.save_fold_models or None)
    if args.refit_full:
        refit_full_models(args.models, args.tasks, args.targets)


if __name__ == "__main__":
    main()
