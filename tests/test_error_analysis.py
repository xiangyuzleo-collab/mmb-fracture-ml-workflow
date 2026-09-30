"""Regression checks for condition-level held-out error summaries."""

from __future__ import annotations

import unittest

import pandas as pd

from src.validation.error_analysis import summarize_condition_errors


class ConditionErrorTests(unittest.TestCase):
    def test_condition_error_metrics_use_held_out_predictions(self) -> None:
        rows = pd.DataFrame(
            {
                "validation_strategy": ["repeated_random_split"] * 3,
                "model": ["ridge_regression"] * 3,
                "direction": ["R", "R", "T"],
                "lever_arm_length": [80, 80, 80],
                "actual": [1.0, 2.0, 1.0],
                "predicted": [1.5, 1.5, 2.0],
            }
        )
        summary = summarize_condition_errors(rows)
        radial = summary.loc[summary["direction"] == "R"].iloc[0]
        self.assertEqual(radial["n_predictions"], 2)
        self.assertAlmostEqual(radial["mean_error"], 0.0)
        self.assertAlmostEqual(radial["MAE"], 0.5)
        self.assertAlmostEqual(radial["RMSE"], 0.5)
        tangential = summary.loc[summary["direction"] == "T"].iloc[0]
        self.assertAlmostEqual(tangential["mean_error"], 1.0)

    def test_missing_columns_and_nonfinite_errors_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing prediction columns"):
            summarize_condition_errors(pd.DataFrame({"actual": [1.0]}))
        rows = pd.DataFrame(
            {
                "validation_strategy": ["repeated_random_split"],
                "model": ["ridge_regression"],
                "direction": ["R"],
                "lever_arm_length": [80],
                "actual": [1.0],
                "predicted": [float("inf")],
            }
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_condition_errors(rows)


if __name__ == "__main__":
    unittest.main()
