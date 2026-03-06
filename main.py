import os
from pathlib import Path
from numbers import Number
from typing import Any

import numpy as np
import pandas as pd

from autoprognosis.explorers.core.defaults import (
    default_feature_scaling_names,
    default_feature_selection_names,
    default_imputers_names,
    default_risk_estimation_names,
)
from autoprognosis.hooks import Hooks
from autoprognosis.studies.risk_estimation import RiskEstimationStudy
from autoprognosis.utils.metrics import evaluate_brier_score, evaluate_c_index
from autoprognosis.utils.serialization import load_model_from_file, save_model_to_file


class PrintHooks(Hooks):
    def cancel(self) -> bool:
        return False

    def heartbeat(self, topic: str, subtopic: str, event_type: str, **kwargs):
        def _fmt(val):
            if isinstance(val, Number):
                return f"{val:.4f}"
            return str(val)

        name = kwargs.get("name", "unknown")
        score = kwargs.get("score")
        c_index = kwargs.get("c_index") or kwargs.get("cindex")
        brier = kwargs.get("brier_score") or kwargs.get("brier")
        aucroc = kwargs.get("aucroc")
        duration = kwargs.get("duration")
        msg = f"[{topic}/{subtopic}] {event_type} -> {name}"
        if score is not None:
            msg += f" | score={_fmt(score)}"
        if c_index is not None:
            msg += f" | c_index={_fmt(c_index)}"
        if aucroc is not None:
            msg += f" | aucroc={_fmt(aucroc)}"
        if brier is not None:
            msg += f" | brier={_fmt(brier)}"
        if duration is not None:
            msg += f" | time={duration:.1f}s"
        print(msg)

    def finish(self) -> None:
        print("Search finished")

data_path = Path(os.environ.get("DATA_PATH", "data/input_data.xlsx"))
sheet_name = "1"

feature_cols = [
    "Age",
    "Sex",
    "Smoke",
    "Histo",
    "TMB",
    "PDL1",
    "Stage",
    "Line",
    "Drug",
    "Treatment",
    "NLR_class"
]
time_col = "OS_Months"
event_col = "OS_Event"
region_col = "region"
dataset_col = "Dataset"

try:
    df = pd.read_excel(data_path, sheet_name=sheet_name)
except ValueError as exc:
    raise ValueError(f"Failed to read Excel; confirm sheet {sheet_name} exists") from exc

total_patients = len(df)
workspace = Path("workspace")

df[dataset_col] = df[dataset_col].astype(str).str.strip()
allowed_dataset_values = {"train", "test1", "test2"}
ds_lower = df[dataset_col].str.lower()
invalid_values = sorted(set(ds_lower.unique()) - allowed_dataset_values)
if invalid_values:
    raise ValueError(f"Unsupported dataset values: {invalid_values}; use Train/Test1/Test2")

train_mask = ds_lower == "train"
test_mask = ds_lower.isin({"test1", "test2"})

if not train_mask.any():
    raise ValueError("No Train samples found; cannot build training set.")
if not test_mask.any():
    raise ValueError("No Test1/Test2 samples found; cannot build test set.")

train_df_full = df[train_mask].copy()
test_df_full = df[test_mask].copy()

train_counts = train_df_full[region_col].value_counts().sort_index()
test_counts = test_df_full[region_col].value_counts().sort_index()
distribution_table = pd.concat(
    [train_counts.rename("Train"), test_counts.rename("Test")], axis=1
).fillna(0).astype(int).reset_index().rename(columns={"index": region_col})

X_train = train_df_full[feature_cols]
T_train = train_df_full[time_col]
Y_train = train_df_full[event_col]

X_test = test_df_full[feature_cols]
T_test = test_df_full[time_col]
Y_test = test_df_full[event_col]

print(
    "\n=== Train/Test Split ===\n"
    f"Train patients: {len(train_df_full)} (region values={train_df_full[region_col].nunique()}), "
    f"Test patients: {len(test_df_full)} (region values={test_df_full[region_col].nunique()}), "
    f"ratio={len(train_df_full) / total_patients:.3f}"
)

train_df = X_train.copy()
train_df[time_col] = T_train
train_df[event_col] = Y_train

raw_horizons = np.linspace(T_train.min(), T_train.max(), 5)[1:-1]
eval_time_horizons = sorted(
    {int(h) for h in raw_horizons if int(h) > 0}
)
if not eval_time_horizons:
    eval_time_horizons = [int(T_train.max())]
def _has_two_classes(df_subset: pd.DataFrame, horizon: int) -> bool:
    labels = (
        (df_subset[event_col] == 1) & (df_subset[time_col] <= horizon)
    ).astype(int)
    return labels.nunique() == 2

eval_time_horizons = [
    h for h in eval_time_horizons if _has_two_classes(train_df, h)
] or [eval_time_horizons[-1]]
full_horizon = int(np.ceil(T_train.max()))
if full_horizon > 0:
    eval_time_horizons = sorted({*eval_time_horizons, full_horizon})

study_name = "ici_risk"

base_risk_estimators = [
    *default_risk_estimation_names,
]
base_imputers = [
    *default_imputers_names,
]
base_feature_scaling = [
    *default_feature_scaling_names,
]
base_feature_selection = [
    *default_feature_selection_names,
]


def _build_study(
    risk_estimators: list[str],
    imputers: list[str],
    feature_scaling: list[str],
    feature_selection: list[str],
) -> RiskEstimationStudy:
    print(
        f"Using risk estimators: {risk_estimators}\n"
        f"Using imputers: {imputers}\n"
        f"Using feature scaling: {feature_scaling}\n"
        f"Using feature selection: {feature_selection}"
    )
    return RiskEstimationStudy(
        study_name=study_name,
        dataset=train_df,
        target=event_col,
        time_to_event=time_col,
        time_horizons=eval_time_horizons,
        num_iter=100,
        num_study_iter=100,
        timeout=120,
        imputers=imputers,
        feature_scaling=feature_scaling,
        feature_selection=feature_selection,
        risk_estimators=risk_estimators,
        score_threshold=0.45,
        workspace=workspace,
        hooks=PrintHooks(),
        n_folds_cv=5,
    )

study = _build_study(
    base_risk_estimators,
    base_imputers,
    base_feature_scaling,
    base_feature_selection,
)

model_arch = study.run()

if model_arch is None:
    raise RuntimeError("No model met the score threshold; lower score_threshold or increase iterations.")

output = workspace / study_name / "model.p"

model = load_model_from_file(output) if output.is_file() else model_arch

model.fit(X_train, T_train, Y_train)
save_model_to_file(output, model)

pred = model.predict(X_test, eval_time_horizons).to_numpy()

print(f"Model architecture: {model.name()}")

for idx, horizon in enumerate(eval_time_horizons):
    capped_test_times = T_test.copy()
    capped_test_times[capped_test_times > T_train.max()] = T_train.max()
    eval_horizon = min(horizon, capped_test_times.max() - 1)

    c_index = evaluate_c_index(
        T_train, Y_train, pred[:, idx], capped_test_times, Y_test, eval_horizon
    )
    brier = evaluate_brier_score(
        T_train, Y_train, pred[:, idx], capped_test_times, Y_test, eval_horizon
    )
    print(f"Horizon {horizon:.2f} months: C-index={c_index:.3f}, Brier={brier:.3f}")
