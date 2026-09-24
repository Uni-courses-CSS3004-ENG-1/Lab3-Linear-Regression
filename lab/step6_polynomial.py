"""Step 6 — Polynomial regression on the best feature, degrees 1 to 5."""
from dataclasses import dataclass

import pandas as pd

from lab.config import DEGREES, MIN_GAIN
from lab.modeling import cv_scores, poly_model
from lab.plots import plot_r2_vs_degree
from lab.report import Report
from lab.step3_simple import SimpleFit


@dataclass(frozen=True)
class PolySearch:
    table: pd.DataFrame   # indexed by degree: train_r2, test_r2, change
    best_degree: int      # degree with the highest mean test R²
    best_r2: float        # that mean test R²


def run(report: Report, fit: SimpleFit) -> PolySearch:
    report.section("Step 6 — Polynomial regression")

    # One row per degree, indexed by the degree itself.
    rows = {}
    for degree in DEGREES:
        scores = cv_scores(poly_model(degree), fit.X, fit.y)
        rows[degree] = {"train_r2": scores["train_r2"].mean(),
                        "test_r2": scores["test_r2"].mean()}
    poly = pd.DataFrame.from_dict(rows, orient="index")
    poly["change"] = poly["test_r2"].diff()

    md_rows = []
    for degree, row in poly.iterrows():
        change = "—" if degree == DEGREES[0] else f"{row.change:+.4f}"
        md_rows.append([degree, f"{row.train_r2:.4f}", f"{row.test_r2:.4f}", change])
        print(f"degree {degree}: train R² = {row.train_r2:.4f}, "
              f"test R² = {row.test_r2:.4f}, test change = {change}")
    report.table(["Degree", "Train R² (5-fold mean)", "Test R² (5-fold mean)",
                  "Test R² change"], md_rows)

    best_degree = poly.test_r2.idxmax()
    # The lowest degree whose test R² is within MIN_GAIN of the best one.
    stop_degree = poly.index[poly.test_r2 >= poly.test_r2.max() - MIN_GAIN][0]
    train_monotonic = bool((poly.train_r2.diff().dropna() >= -1e-12).all())
    print(f"Highest mean test R² at degree {best_degree}")
    print(f"Test R² stops improving (within {MIN_GAIN} of the best) at degree {stop_degree}")
    print("Train R² never decreases with degree:", train_monotonic)
    report.add(f"Highest mean test R²: degree **{best_degree}**", "")

    plot_r2_vs_degree(poly, fit.best)
    _write_answers(report, poly, stop_degree, train_monotonic, fit)
    return PolySearch(poly, best_degree, poly.test_r2.max())


def _write_answers(report: Report, poly: pd.DataFrame, stop_degree: int,
                   train_monotonic: bool, fit: SimpleFit) -> None:
    first_degree, max_degree = DEGREES[0], DEGREES[-1]
    later = poly.test_r2[poly.index > stop_degree]  # degrees past the stopping point

    where = ["**Where test R² stops improving.**"]
    if later.empty:
        where.append(f"Degree {max_degree} beats every lower degree by more than {MIN_GAIN}, "
                     f"so there is no clear stopping point within degrees 1–{max_degree} "
                     "on this split.")
    else:
        start_r2 = poly.test_r2[first_degree]
        if stop_degree == first_degree:
            rise = (f"Test R² is {start_r2:.4f} at degree 1 and does not improve "
                    "meaningfully after that")
        else:
            rise = (f"Test R² rises from {start_r2:.4f} at degree 1 to "
                    f"{poly.test_r2[stop_degree]:.4f} at degree {stop_degree} "
                    "and stops improving there")
        where.append(rise + f": no higher degree beats it by more than {MIN_GAIN} ("
                     + ", ".join(f"degree {d}: {r2:.4f}" for d, r2 in later.items()) + ").")
    report.paragraph(*where)

    best = fit.best
    why = [
        "**Why train R² keeps climbing while test R² does not.**",
        f"Train R² goes from {poly.train_r2[first_degree]:.4f} at degree 1 to "
        f"{poly.train_r2[max_degree]:.4f} at degree {max_degree}"
        + (" and never decreases." if train_monotonic else "."),
        "A higher-degree polynomial contains every lower-degree one (set the extra coefficients "
        "to zero), so on the cars it is fitted to it can only match or beat the lower degree.",
        "Each extra term is extra freedom to bend the curve toward the particular training cars, "
        "including their random scatter.",
    ]
    if not later.empty:
        why.append(f"That scatter does not repeat in the held-out fold, so past degree "
                   f"{stop_degree} the extra bends buy nothing on test data, or lose a little.")
    if stop_degree == 2:
        # Describe the curve from the signs of the fitted degree-2 coefficients.
        slope, curvature = poly_model(2).fit(fit.X, fit.y)[-1].coef_
        direction = "falls" if slope < 0 else "rises"
        bend = "flattens out" if slope * curvature < 0 else "gets steeper"
        why.append(f"Degree 2 already captures the real shape (fuel efficiency {direction} as "
                   f"`{best}` grows, and the curve {bend} at high `{best}`), so the higher "
                   "degrees only fit noise.")
    low, high = fit.X[best].min(), fit.X[best].max()
    why.append(f"This relies on standardizing `{best}` inside the pipeline first: its raw values "
               f"span {low:g}–{high:g}, so the degree-{max_degree} term would reach about "
               f"{high ** max_degree:.1e}, which makes plain least squares numerically unstable.")
    report.paragraph(*why)
