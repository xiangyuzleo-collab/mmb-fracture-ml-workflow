from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


@dataclass
class ModelSpec:
    name: str
    family: str
    estimator: Any
    hyperparameters: dict[str, Any]
    optional_dependency_available: bool = True


def _numeric_pipeline(estimator, scale: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_model_specs(model_cfg: dict, random_seed: int = 42, only: list[str] | None = None) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for name, cfg in model_cfg["models"].items():
        if only and name not in only:
            continue
        if not cfg.get("enabled", True):
            continue
        optional = cfg.get("optional_dependency")
        if optional and importlib.util.find_spec(optional) is None:
            specs.append(
                ModelSpec(
                    name=name,
                    family=cfg["family"],
                    estimator=None,
                    hyperparameters=cfg.get("hyperparameters", {}),
                    optional_dependency_available=False,
                )
            )
            continue
        hp = dict(cfg.get("hyperparameters", {}))
        estimator = None
        if name == "mean_predictor":
            estimator = DummyRegressor(strategy="mean")
        elif name == "linear_regression":
            estimator = _numeric_pipeline(LinearRegression())
        elif name == "polynomial_ridge":
            estimator = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("poly", PolynomialFeatures(degree=int(hp.get("degree", 2)), include_bias=False)),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=float(hp.get("alpha", 1.0)))),
                ]
            )
        elif name == "ridge_regression":
            estimator = _numeric_pipeline(Ridge(alpha=float(hp.get("alpha", 1.0))))
        elif name == "lasso_regression":
            estimator = _numeric_pipeline(
                Lasso(alpha=float(hp.get("alpha", 0.002)), max_iter=int(hp.get("max_iter", 20000)))
            )
        elif name == "svr":
            svr = SVR(
                kernel=hp.get("kernel", "rbf"),
                C=float(hp.get("C", 10.0)),
                epsilon=float(hp.get("epsilon", 0.03)),
                gamma=hp.get("gamma", "scale"),
            )
            estimator = _numeric_pipeline(TransformedTargetRegressor(regressor=svr, transformer=StandardScaler()))
        elif name == "random_forest":
            estimator = _numeric_pipeline(
                RandomForestRegressor(
                    n_estimators=int(hp.get("n_estimators", 300)),
                    max_features=hp.get("max_features", "sqrt"),
                    min_samples_leaf=int(hp.get("min_samples_leaf", 2)),
                    random_state=random_seed,
                    n_jobs=1,
                ),
                scale=False,
            )
        elif name == "extra_trees":
            estimator = _numeric_pipeline(
                ExtraTreesRegressor(
                    n_estimators=int(hp.get("n_estimators", 400)),
                    max_features=hp.get("max_features", "sqrt"),
                    min_samples_leaf=int(hp.get("min_samples_leaf", 1)),
                    random_state=random_seed,
                    n_jobs=1,
                ),
                scale=False,
            )
        elif name == "gradient_boosting":
            estimator = _numeric_pipeline(
                GradientBoostingRegressor(
                    n_estimators=int(hp.get("n_estimators", 180)),
                    learning_rate=float(hp.get("learning_rate", 0.04)),
                    max_depth=int(hp.get("max_depth", 2)),
                    random_state=random_seed,
                ),
                scale=False,
            )
        elif name == "xgboost":
            from xgboost import XGBRegressor

            estimator = _numeric_pipeline(
                XGBRegressor(
                    n_estimators=int(hp.get("n_estimators", 240)),
                    max_depth=int(hp.get("max_depth", 3)),
                    learning_rate=float(hp.get("learning_rate", 0.04)),
                    subsample=float(hp.get("subsample", 0.85)),
                    colsample_bytree=float(hp.get("colsample_bytree", 0.85)),
                    reg_lambda=float(hp.get("reg_lambda", 2.0)),
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=random_seed,
                    n_jobs=1,
                ),
                scale=False,
            )
        elif name == "gaussian_process":
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0) + WhiteKernel(
                noise_level=1e-3
            )
            estimator = _numeric_pipeline(
                TransformedTargetRegressor(
                    regressor=GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=random_seed),
                    transformer=StandardScaler(),
                )
            )
        elif name == "mlp":
            hidden = tuple(int(x) for x in hp.get("hidden_layer_sizes", [32, 16]))
            estimator = _numeric_pipeline(
                TransformedTargetRegressor(
                    regressor=MLPRegressor(
                        hidden_layer_sizes=hidden,
                        alpha=float(hp.get("alpha", 0.001)),
                        learning_rate_init=float(hp.get("learning_rate_init", 0.001)),
                        max_iter=int(hp.get("max_iter", 1500)),
                        early_stopping=bool(hp.get("early_stopping", True)),
                        random_state=random_seed,
                    ),
                    transformer=StandardScaler(),
                )
            )
        if estimator is not None:
            specs.append(ModelSpec(name=name, family=cfg["family"], estimator=estimator, hyperparameters=hp))
    return specs
