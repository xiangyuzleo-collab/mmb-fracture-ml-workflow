"""Check that specimen-level validation cannot split repeated specimens."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models.evaluate_models import run_validation
from src.utils.paths import project_root
from src.validation.cross_validation_utils import require_specimen_level_rows
from src.validation.leave_one_lever_arm_out import leave_one_lever_arm_splits
from src.validation.repeated_random_split import repeated_random_splits


class SpecimenValidationTests(unittest.TestCase):
    def test_missing_and_duplicate_specimen_ids_are_rejected(self) -> None:
        for ids in (["SYN-001", "SYN-001"], ["SYN-001", ""], ["SYN-001", None]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                require_specimen_level_rows(pd.DataFrame({"specimen_id": ids}))

    def test_synthetic_splits_hold_out_whole_specimens(self) -> None:
        df = pd.read_csv(project_root() / "examples" / "synthetic_specimens.csv")
        require_specimen_level_rows(df)
        strata = df["direction"] + "_c" + df["lever_arm_length"].astype(str)
        splits = list(repeated_random_splits(df.index, 3, 0.2, 42, strata))
        splits.extend(leave_one_lever_arm_splits(df))
        self.assertEqual(len(splits), 6)
        for split in splits:
            with self.subTest(fold=split["fold"]):
                train = set(df.loc[split["train_indices"], "specimen_id"])
                test = set(df.loc[split["test_indices"], "specimen_id"])
                self.assertFalse(train & test)
                self.assertEqual(len(train | test), len(df))

    def test_validation_stops_before_writing_outputs_for_duplicate_ids(self) -> None:
        df = pd.read_csv(project_root() / "examples" / "synthetic_specimens.csv")
        duplicate = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "duplicate_specimens.csv"
            duplicate.to_csv(dataset, index=False)
            output = root / "outputs"
            with self.assertRaisesRegex(ValueError, "one row per specimen_id"):
                run_validation("repeated_random_split", dataset_path=dataset, output_root=output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
