from __future__ import annotations

from pathlib import Path

from src.data.clean_data import build_clean_datasets
from src.data.compute_mmb_parameters import compute_specimen_parameters
from src.features.build_features import build_feature_manifest


def build_dataset() -> dict[str, Path]:
    """Build cleaned point data, specimen-level parameters and feature manifest.

    This module exists as the canonical dataset-construction entry point for the
    project structure. It delegates to the narrower processing modules so that
    raw-data cleaning, MMB parameter extraction and feature manifest generation
    remain independently runnable.
    """
    outputs: dict[str, Path] = {}
    outputs.update(build_clean_datasets())
    outputs.update(compute_specimen_parameters())
    outputs["feature_manifest"] = build_feature_manifest()
    return outputs


if __name__ == "__main__":
    build_dataset()
