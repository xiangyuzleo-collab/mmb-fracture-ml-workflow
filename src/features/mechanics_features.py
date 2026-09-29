from __future__ import annotations

import numpy as np
import pandas as pd


def add_mechanics_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lever_arm_over_thickness"] = out["lever_arm_length"] / out["specimen_thickness"].replace(0, np.nan)
    out["lever_arm_over_span"] = out["lever_arm_length"] / out["span_length"].replace(0, np.nan)
    out["initial_crack_over_length"] = out["initial_crack_length"] / out["span_length"].replace(0, np.nan)
    out["initial_crack_over_thickness"] = out["initial_crack_length"] / out["specimen_thickness"].replace(0, np.nan)
    out["width_over_thickness"] = out["specimen_width"] / out["specimen_thickness"].replace(0, np.nan)
    out["height_over_thickness"] = out["specimen_height"] / out["specimen_thickness"].replace(0, np.nan)
    out["direction_R"] = (out["direction"] == "R").astype(int)
    out["direction_T"] = (out["direction"] == "T").astype(int)
    return out
