# MMB fracture machine learning workflow

Python code adapted from a research workflow for mixed-mode bending (MMB)
fracture experiments on engineered bamboo. The repository shows the data
processing, feature design, model comparison and validation logic. It contains
**no original specimen measurements, unpublished result tables or manuscript**.
The included CSV is entirely artificial and is only for checking that the
software runs. Its metrics have no scientific meaning.

## What the code does

1. `src/data/` reads a user-supplied workbook, standardizes columns and derives
   specimen-level response tables.
2. `src/features/` forms geometry and mechanics features and excludes direct
   target and configured target-derived variables.
3. `src/models/` defines mean, linear, ridge, SVR, tree, Gaussian-process and
   MLP candidates. Imputation and scaling are fitted within training folds.
4. `src/validation/` supplies repeated random splits and leave-one-lever-arm-
   length-out splits. `src/models/evaluate_models.py` records fold predictions,
   metrics and split indices.
5. `src/interpretation/` and `src/visualization/` contain downstream analysis
   and plotting code. SHAP and XGBoost are optional.

The validation unit is a specimen. The source research workflow has additional
checks and unpublished results that are intentionally absent here. This public
code does not claim a validated predictor for unseen loading configurations or
an experimentally verified optimum.
Validation requires one row and one non-empty `specimen_id` per specimen; it
rejects duplicate IDs to prevent the same specimen entering both sides of a
row-based train/test split.

## Run the artificial-data demo

Use Python 3.10 from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_synthetic_demo.py
```

The command uses `examples/synthetic_specimens.csv` and runs three models on
both validation strategies. It writes CSV metrics, predictions and split
records under `outputs/synthetic_demo/`. The directory is ignored by Git.
This command does not need any private experiment files.

To run the specimen-split checks against the public synthetic data:

```bash
.venv/bin/python -m unittest discover -s tests
```

## Use your own experimental data

The numbered scripts in `scripts/` retain the original research pipeline.
Place data you are authorized to process under `data/raw/`, adapt
`config/config.yaml` to the local workbook names and sheets, and check the
field mapping in `src/data/clean_data.py`. The source workflow expects a main
workbook with pointwise and specimen-parameter sheets plus a separate
correction-summary workbook. Missing correction matches fail explicitly.
Run the scripts from the repository root in number order; begin with
`00_inspect_raw_data.py` and inspect each intermediate output before training.
No original workbook is supplied, so the full experimental pipeline is **not**
reproducible from this repository alone.

Data provenance, rights and target definitions must be checked for each new
dataset. Response-derived features in tasks B and C are for post-test analysis;
they must not be described as predictions available before an experiment.
The raw-data processing code has not been re-run against a releasable dataset
in this copy. The artificial demo exercises the model-validation path only.

## Repository boundaries

- `examples/` contains deliberately synthetic specimen rows labeled
  `synthetic_demo`.
- `data/raw/`, `data/interim/`, `data/processed/` and `outputs/` are excluded
  from Git. Do not change that exclusion to publish experimental measurements.
- Empty explanatory notebooks, local model binaries, plots, draft papers and
  unpublished result reports from the source directory were not copied.
- The source project had no Git history; this repository does not imply earlier
  commits. The code included here is published with the owner's authorization.

No license is provided. Public visibility does not grant permission to reuse
the code.
