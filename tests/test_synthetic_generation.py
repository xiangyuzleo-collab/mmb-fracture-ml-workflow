"""The committed public example must match its documented generator."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.utils.paths import project_root


class SyntheticGenerationTests(unittest.TestCase):
    def test_committed_example_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generated_path = Path(temporary) / "synthetic_specimens.csv"
            subprocess.run(
                [sys.executable, "scripts/generate_synthetic_data.py", "--output", str(generated_path)],
                cwd=project_root(), capture_output=True, text=True, check=True,
            )
            expected = pd.read_csv(generated_path)
        actual = pd.read_csv(project_root() / "examples" / "synthetic_specimens.csv")
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(len(actual), 36)
        self.assertTrue(actual["sample_type"].eq("synthetic_demo").all())


if __name__ == "__main__":
    unittest.main()
