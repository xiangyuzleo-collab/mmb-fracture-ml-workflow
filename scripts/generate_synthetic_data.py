"""Create an entirely artificial specimen table for exercising public code."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from src.data.synthetic_checks import validate_synthetic_specimens
from src.utils.paths import project_root


SEED = 20260929


def generate_synthetic_specimens(seed: int = SEED) -> pd.DataFrame:
    """Use arbitrary toy ranges and a nonphysical target equation."""
    rng = np.random.default_rng(seed)
    rows = []
    specimen_number = 0
    for lever in (80, 140, 200):
        for direction in ("R", "T"):
            for _ in range(6):
                specimen_number += 1
                thickness = round(float(rng.uniform(17.4, 18.6)), 3)
                density = round(float(rng.uniform(1.00, 1.10)), 3)
                target = (
                    0.78
                    + 0.004 * lever
                    + (0.24 if direction == "T" else 0.0)
                    + 0.15 * (density - 1.05)
                    + float(rng.normal(0, 0.035))
                )
                rows.append(
                    {
                        "specimen_id": f"SYN-{specimen_number:03d}",
                        "group": f"SYN-{direction}-{lever}",
                        "direction": direction,
                        "lever_arm_length": lever,
                        "span_length": round(float(rng.uniform(335, 345)), 3),
                        "specimen_width": round(float(rng.uniform(16.4, 17.6)), 3),
                        "specimen_thickness": thickness,
                        "specimen_height": thickness,
                        "initial_crack_length": round(float(rng.uniform(33, 37)), 3),
                        "density": density,
                        "moisture_content": round(float(rng.uniform(7, 9)), 3),
                        "loading_rate": round(float(rng.uniform(0.9, 1.1)), 3),
                        "direction_R": int(direction == "R"),
                        "direction_T": int(direction == "T"),
                        "E1_MPa": round(float(rng.uniform(4700, 5300)), 2),
                        "E2_MPa": round(float(rng.uniform(3400, 4000)), 2),
                        "E3_MPa": round(float(rng.uniform(2300, 2700)), 2),
                        "G12_MPa": round(float(rng.uniform(620, 780)), 2),
                        "G13_MPa": round(float(rng.uniform(480, 620)), 2),
                        "Gc_raw": round(target, 5),
                        "sample_type": "synthetic_demo",
                    }
                )
    specimens = pd.DataFrame(rows)
    validate_synthetic_specimens(specimens)
    return specimens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=project_root() / "examples" / "synthetic_specimens.csv")
    args = parser.parse_args()
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_specimens().to_csv(path, index=False)
    print(f"Generated {path} from seed {SEED}")


if __name__ == "__main__":
    main()
