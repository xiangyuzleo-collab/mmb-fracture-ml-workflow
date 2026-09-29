from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from src.features.build_features import load_modeling_dataset
from src.models.model_selection import build_model_ranking
from src.utils.paths import outputs_dir


warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _model_registry() -> pd.DataFrame:
    path = outputs_dir() / "models" / "model_registry.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _best_row(preferred_target: str = "Gc_raw") -> pd.Series | None:
    ranking_path = outputs_dir() / "tables" / "model_comparison" / "final_model_ranking.xlsx"
    if not ranking_path.exists():
        build_model_ranking()
    ranking = pd.read_excel(ranking_path, sheet_name="final_model_ranking")
    eligible = ranking.get("eligible_for_surrogate", pd.Series(False, index=ranking.index)).fillna(False).astype(bool)
    sub = ranking[(ranking["target"] == preferred_target) & eligible].sort_values("composite_score", ascending=False)
    if sub.empty:
        sub = ranking[eligible].sort_values("composite_score", ascending=False)
    return sub.iloc[0] if not sub.empty else None


def run_permutation_importance() -> dict[str, Path]:
    best = _best_row()
    out_dir = outputs_dir() / "figures" / "feature_importance"
    out_dir.mkdir(parents=True, exist_ok=True)
    if best is None:
        status = outputs_dir() / "reports" / "interpretability_status.md"
        status.write_text(
            "# Interpretability Status\n\nStatus: **BLOCKED_NO_VALIDATED_SURROGATE**\n\n"
            "No permutation-importance result is published because no model passed the leave-one-lever-arm generalization gate.\n",
            encoding="utf-8",
        )
        return {}
    registry = _model_registry()
    match = registry[(registry["task"] == best["task"]) & (registry["target"] == best["target"]) & (registry["model"] == best["model"])]
    if match.empty:
        return {}
    model_path = Path(match.iloc[0]["model_path"])
    metadata_path = Path(match.iloc[0]["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = joblib.load(model_path)
    df = load_modeling_dataset()
    features = metadata["input_features"]
    target = metadata["target_variable"]
    data = df[features + [target]].dropna(subset=[target])
    result = permutation_importance(
        model,
        data[features].to_numpy(),
        data[target].to_numpy(),
        n_repeats=50,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    imp = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "task": metadata["task_name"],
            "target": target,
            "model": metadata["model_name"],
        }
    ).sort_values("importance_mean", ascending=False)
    csv_path = out_dir / "permutation_importance_best_model_data.csv"
    imp.to_csv(csv_path, index=False)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    top = imp.head(15).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#8fb3c7", edgecolor="black")
    ax.set_xlabel("Permutation importance (RMSE increase)")
    ax.set_title(f"{metadata['model_name']} | {metadata['task_name']} | {target}")
    fig.tight_layout()
    png = out_dir / "permutation_importance_best_model.png"
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "permutation_importance_best_model.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"csv": csv_path, "png": png}


if __name__ == "__main__":
    run_permutation_importance()
