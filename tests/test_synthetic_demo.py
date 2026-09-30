"""The public demo must be self-contained and describe its own run."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest

import pandas as pd

from src.data.synthetic_checks import validate_synthetic_specimens
from src.utils.io_utils import sha256_file
from src.utils.paths import project_root


class SyntheticDemoTests(unittest.TestCase):
    def test_demo_rejects_rows_without_synthetic_labels(self) -> None:
        for ids, labels in ((["SYN-001"], ["experiment"]), (["REAL-001"], ["synthetic_demo"])):
            with self.subTest(ids=ids, labels=labels), self.assertRaises(ValueError):
                validate_synthetic_specimens(pd.DataFrame({"specimen_id": ids, "sample_type": labels}))

    def test_demo_writes_a_reproducibility_manifest(self) -> None:
        root = project_root()
        subprocess.run(
            [sys.executable, "scripts/run_synthetic_demo.py"],
            cwd=root, capture_output=True, text=True, check=True,
        )
        manifest = json.loads((root / "outputs" / "synthetic_demo" / "run_manifest.json").read_text())
        self.assertEqual(manifest["dataset"], "examples/synthetic_specimens.csv")
        self.assertEqual(manifest["dataset_sha256"], sha256_file(root / manifest["dataset"]))
        self.assertEqual(manifest["specimen_count"], 36)
        self.assertEqual(manifest["sample_type"], "synthetic_demo")
        self.assertEqual(len(manifest["outputs"]), 2)
        for run in manifest["outputs"]:
            self.assertGreater(run["metric_rows"], 0)
            for key in ("metrics", "predictions", "split_summary", "condition_errors"):
                self.assertTrue(run[key].startswith("outputs/synthetic_demo/"))
                self.assertTrue((root / run[key]).exists())


if __name__ == "__main__":
    unittest.main()
