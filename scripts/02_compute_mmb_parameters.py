from __future__ import annotations

import _bootstrap  # noqa: F401
from src.data.compute_mmb_parameters import compute_specimen_parameters
from src.visualization.plotting import experimental_figures


if __name__ == "__main__":
    compute_specimen_parameters()
    experimental_figures()
