# ICI Survival Risk Modeling (code companion)

This package contains the code used to build and evaluate the survival risk model in the manuscript. Patient-level data are not distributed for ethical reasons.

## Contents
- `main.py`: end-to-end training script using AutoPrognosis survival pipelines (default search space for imputers, feature scaling, feature selection, and risk estimators).
- `src/`, `scripts/`, `tests/`, `third_party/`: supporting library code from AutoPrognosis.
- Output model is written to `workspace/<study_name>/model.p` after running `main.py`.

## Setup
1) Install dependencies (Python 3.8+ recommended):
   ```bash
   pip install -e .
   ```
2) Provide an input spreadsheet via environment variable `DATA_PATH` (default `data/input_data.xlsx`) containing:
   - Features: `Age, Sex, Smoke, Histo, TMB, PDL1, Stage, Line, Drug, Treatment, NLR_class`
   - Survival fields: `OS_Months` (time), `OS_Event` (event)
   - Metadata: `region`, `Dataset` with values `train`, `test1`, `test2`
3) Run:
   ```bash
   export DATA_PATH=/path/to/your/data.xlsx
   python main.py
   ```

## Customization
- To adjust the search space, edit the `base_*` lists in `main.py` (risk estimators, imputers, feature scaling, feature selection).

## License
Apache 2.0 (inherits upstream AutoPrognosis license).
