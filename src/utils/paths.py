from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return project_root() / "config"


def data_dir() -> Path:
    return project_root() / "data"


def outputs_dir() -> Path:
    return project_root() / "outputs"


def docs_dir() -> Path:
    return project_root() / "docs"


def manuscript_dir() -> Path:
    return project_root() / "manuscript"


def ensure_project_dirs() -> None:
    for path in [
        data_dir() / "raw",
        data_dir() / "interim",
        data_dir() / "processed",
        data_dir() / "external",
        docs_dir(),
        outputs_dir() / "logs",
        outputs_dir() / "reports",
    ]:
        path.mkdir(parents=True, exist_ok=True)
