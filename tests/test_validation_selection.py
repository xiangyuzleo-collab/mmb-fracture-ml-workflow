"""Invalid run selections should fail clearly instead of producing empty reports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.models.evaluate_models import run_validation
from src.utils.paths import project_root


class ValidationSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = project_root() / "examples" / "synthetic_specimens.csv"

    def test_unknown_model_task_and_target_fail_before_outputs(self) -> None:
        selections = (
            ({"only_models": ["not_a_model"]}, "Unknown models"),
            ({"only_tasks": ["not_a_task"]}, "Unknown tasks"),
            ({"only_targets": ["not_a_target"]}, "Unknown targets"),
        )
        for selected, error in selections:
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "outputs"
                with self.assertRaisesRegex(ValueError, error):
                    run_validation(
                        "repeated_random_split", dataset_path=self.dataset,
                        output_root=output, **selected,
                    )
                self.assertFalse(output.exists())

    def test_empty_selection_fails_instead_of_running_every_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "outputs"
            with self.assertRaisesRegex(ValueError, "No models selected"):
                run_validation(
                    "repeated_random_split", only_models=[],
                    dataset_path=self.dataset, output_root=output,
                )
            self.assertFalse(output.exists())

    def test_unavailable_target_does_not_leave_partial_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "outputs"
            with self.assertRaisesRegex(RuntimeError, "no metrics"):
                run_validation(
                    "repeated_random_split", only_models=["mean_predictor"],
                    only_tasks=["task_A_pre_test"], only_targets=["Pmax"],
                    dataset_path=self.dataset, output_root=output,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
