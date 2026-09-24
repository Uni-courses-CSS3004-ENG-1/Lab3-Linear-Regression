"""Step 4 — Evaluate the simple model honestly: train/test split, then 5-fold CV."""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from lab.config import RANDOM_STATE
from lab.modeling import cv_scores, regression_metrics
from lab.report import Report
from lab.step3_simple import SimpleFit


@dataclass(frozen=True)
class SimpleScores:
    cv_mean: float   # mean test R² over the 5 folds
    cv_std: float
    cv_rmse: float   # mean test RMSE over the 5 folds


def run(report: Report, fit: SimpleFit) -> SimpleScores:
    report.section("Step 4 — Evaluate the model honestly")

    # --- one train/test split: fit on the training rows only ---
    X_train, X_test, y_train, y_test = train_test_split(
        fit.X, fit.y, test_size=0.2, random_state=RANDOM_STATE
    )
    n_train, n_test = len(X_train), len(X_test)
    shared_rows = len(X_train.index.intersection(X_test.index))
    split_model = LinearRegression().fit(X_train, y_train)
    train_m = regression_metrics(y_train, split_model.predict(X_train))
    test_m = regression_metrics(y_test, split_model.predict(X_test))

    print(f"random_state = {RANDOM_STATE}; train rows = {n_train}, test rows = {n_test}, "
          f"rows in both = {shared_rows}")
    print(f"{'Metric':<6}{'Train':>12}{'Test':>12}")
    for name in train_m:
        print(f"{name:<6}{train_m[name]:>12.4f}{test_m[name]:>12.4f}")
    report.table(["Metric", f"Train ({n_train} rows)", f"Test ({n_test} rows)"],
                 [[name, f"{train_m[name]:.4f}", f"{test_m[name]:.4f}"] for name in train_m])

    train_better = (train_m["R²"] > test_m["R²"]) and (train_m["RMSE"] < test_m["RMSE"])
    print("Checkpoint: training metrics better than test metrics?", train_better)

    # --- 5-fold cross-validation ---
    scores = cv_scores(LinearRegression(), fit.X, fit.y)
    folds = scores["test_r2"]
    result = SimpleScores(folds.mean(), folds.std(), scores["test_rmse"].mean())
    print("\n5-fold CV test R² per fold:", np.round(folds, 4))
    print(f"Mean test R² = {result.cv_mean:.4f}, std = {result.cv_std:.4f}")
    report.add(f"Checkpoint — train metrics better than test metrics: **{train_better}**", "")
    report.table(["5-fold CV", *[f"Fold {i}" for i in range(1, 6)], "Mean", "Std"],
                 [["Test R²", *[f"{s:.4f}" for s in folds],
                   f"{result.cv_mean:.4f}", f"{result.cv_std:.4f}"]])

    # --- written answer ---
    if train_better:
        checkpoint_note = [
            f"Here train R² ({train_m['R²']:.4f}) is above test R² ({test_m['R²']:.4f}), as the "
            f"checkpoint expects, but only by {train_m['R²'] - test_m['R²']:.4f}.",
        ]
        luck = "the size of the gap mostly reflects"
    else:
        checkpoint_note = [
            f"Here test R² ({test_m['R²']:.4f}) is actually above train R² ({train_m['R²']:.4f}).",
            f"That is not a leak: the model was fitted on the training rows only, and the two "
            f"sets share {shared_rows} rows.",
        ]
        luck = "which set scores higher mostly depends on"
    checkpoint_note.append(f"A straight line has only two parameters and cannot memorise "
                           f"{n_train} cars, so {luck} which {n_test} cars landed in the test set.")
    report.paragraph(
        "**Why the training score alone is not enough.**",
        "The training score is measured on the same cars the model used to choose θ0 and θ1, so it",
        "only shows how well the line fits data it has already seen; a flexible enough model could",
        "memorise the training cars, score perfectly on them and still fail on a new one.",
        "Only cars held out from fitting (the test set, or the held-out fold in cross-validation)",
        "show how the model does on a car it hasn't seen.",
        *checkpoint_note,
        f"This is why the 5-fold result (mean test R² {result.cv_mean:.4f}, "
        f"std {result.cv_std:.4f}),",
        "which averages over five different splits, is the more reliable estimate.",
    )
    return result
