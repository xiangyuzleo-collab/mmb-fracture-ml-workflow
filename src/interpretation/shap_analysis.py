from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import load_modeling_dataset
from src.interpretation.permutation_importance_analysis import _best_row
from src.models.model_selection import build_model_ranking
from src.utils.paths import outputs_dir


def run_shap_analysis() -> dict[str, Path]:
    out_dir = outputs_dir() / "figures" / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = outputs_dir() / "reports" / "shap_analysis_report.md"
    try:
        import shap
    except Exception as exc:
        report.write_text(
            "# SHAP Analysis Report\n\nSHAP is unavailable in the current environment. Permutation importance is used as the primary model-agnostic explanation method.\n\n"
            f"Import error: `{exc}`\n",
            encoding="utf-8",
        )
        return {"report": report}

    try:
        best = _best_row()
        if best is None:
            build_model_ranking()
            best = _best_row()
        if best is None:
            report.write_text(
                "# SHAP Analysis Report\n\nStatus: **BLOCKED_NO_VALIDATED_SURROGATE**\n\n"
                "SHAP is not published because no model passed the leave-one-lever-arm generalization gate. "
                "Explaining a screening-only model as the final surrogate would be misleading.\n",
                encoding="utf-8",
            )
            return {"report": report}
        registry = pd.read_csv(outputs_dir() / "models" / "model_registry.csv")
        match = registry[(registry["task"] == best["task"]) & (registry["target"] == best["target"]) & (registry["model"] == best["model"])]
        if match.empty:
            raise RuntimeError("Selected best model is missing from model_registry.csv.")
        metadata = json.loads(Path(match.iloc[0]["metadata_path"]).read_text(encoding="utf-8"))
        model = joblib.load(match.iloc[0]["model_path"])
        features = metadata["input_features"]
        target = metadata["target_variable"]
        df = load_modeling_dataset()
        data = df[features + [target]].dropna(subset=[target]).reset_index(drop=True)
        X_df = data[features]
        X = X_df.to_numpy()
        background = X[: min(20, len(X))]

        def predict_fn(values):
            return np.asarray(model.predict(pd.DataFrame(values, columns=features)), dtype=float)

        explainer = shap.Explainer(predict_fn, background, feature_names=features)
        values = explainer(X)
        shap_values = np.asarray(values.values, dtype=float)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]
        summary = pd.DataFrame(
            {
                "feature": features,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                "task": metadata["task_name"],
                "target": target,
                "model": metadata["model_name"],
            }
        ).sort_values("mean_abs_shap", ascending=False)
        summary_csv = out_dir / "shap_summary_best_model_data.csv"
        summary.to_csv(summary_csv, index=False)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        top = summary.head(15).iloc[::-1]
        ax.barh(top["feature"], top["mean_abs_shap"], color="#b9a4d0", edgecolor="black")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"SHAP summary | {metadata['model_name']} | {target}")
        fig.tight_layout()
        png_summary = out_dir / "shap_summary_best_model.png"
        fig.savefig(png_summary, bbox_inches="tight", dpi=300)
        fig.savefig(out_dir / "shap_summary_best_model.pdf", bbox_inches="tight")
        plt.close(fig)

        dependence_feature = "lever_arm_length" if "lever_arm_length" in features else features[0]
        j = features.index(dependence_feature)
        dep = pd.DataFrame(
            {
                dependence_feature: X_df[dependence_feature],
                "shap_value": shap_values[:, j],
                "target": target,
                "model": metadata["model_name"],
            }
        )
        dep_csv = out_dir / "shap_dependence_lever_arm_data.csv"
        dep.to_csv(dep_csv, index=False)
        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        ax.scatter(dep[dependence_feature], dep["shap_value"], edgecolor="black", alpha=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(dependence_feature)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP dependence | {dependence_feature}")
        png_dep = out_dir / "shap_dependence_lever_arm.png"
        fig.savefig(png_dep, bbox_inches="tight", dpi=300)
        fig.savefig(out_dir / "shap_dependence_lever_arm.pdf", bbox_inches="tight")
        plt.close(fig)

        report.write_text(
            "# SHAP Analysis Report\n\n"
            f"SHAP was computed for `{metadata['model_name']}` on `{metadata['task_name']}` with target `{target}`. "
            "Permutation importance remains the primary model-agnostic explanation, while SHAP is provided as supplementary local-additive interpretation.\n",
            encoding="utf-8",
        )
        return {"report": report, "summary": summary_csv, "summary_png": png_summary, "dependence": dep_csv, "dependence_png": png_dep}
    except Exception as exc:
        report.write_text(
            "# SHAP Analysis Report\n\n"
            "SHAP was available but the generic SHAP calculation failed. Permutation importance remains the primary explanation output.\n\n"
            f"Error: `{exc}`\n",
            encoding="utf-8",
        )
        return {"report": report}


if __name__ == "__main__":
    run_shap_analysis()
