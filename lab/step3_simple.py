"""Step 3 — Simple linear regression on the most correlated feature."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from lab.config import TARGET
from lab.plots import plot_simple_fit
from lab.report import Report


@dataclass(frozen=True)
class SimpleFit:
    best: str                    # the most correlated feature
    numeric_features: list[str]  # all numeric features, best-correlated first
    X: pd.DataFrame              # df[[best]]
    y: pd.Series                 # the target


def run(report: Report, df: pd.DataFrame) -> SimpleFit:
    report.section("Step 3 — Simple linear regression")

    # Rank features by |correlation|: a strong negative link is as useful as a positive one.
    corr = df.corr(numeric_only=True)[TARGET].drop(TARGET)
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    print("Correlation of each numeric feature with the target:")
    print(corr.round(4))
    report.table(["Feature", f"Correlation with `{TARGET}`"],
                 [[f"`{name}`", f"{value:+.4f}"] for name, value in corr.items()])

    numeric_features = list(corr.index)
    best = numeric_features[0]
    print(f"\nMost strongly correlated feature: {best} (r = {corr[best]:+.4f})")

    X = df[[best]]
    y = df[TARGET]

    model = LinearRegression()
    model.fit(X, y)          # X: shape (n, 1); y: shape (n,)
    print(model.intercept_, model.coef_)
    theta0, theta1 = model.intercept_, model.coef_[0]
    equation = f"y = {theta0:.4f} + ({theta1:.6f}) * x"
    plot_simple_fit(X, y, best, model, equation)

    # The value to predict for: a round number near the middle of the feature's range.
    median = X[best].median()
    x_new = float(np.round(median, -1 if median >= 50 else 0))
    y_new = model.predict(pd.DataFrame({best: [x_new]}))[0]
    print(f"θ0 (intercept) = {theta0:.4f}")
    print(f"θ1 (slope)     = {theta1:.6f}")
    print(f"Fitted equation: {equation}")
    print(f"Prediction for {best} = {x_new:g}: "
          f"y = {theta0:.4f} + ({theta1:.6f}) * {x_new:g} = {y_new:.2f} km/L")
    report.table(["Feature", "θ0 (intercept)", "θ1 (slope)", "Example x", "Predicted y (km/L)"],
                 [[f"`{best}`", f"{theta0:.4f}", f"{theta1:.6f}", f"{x_new:g}", f"{y_new:.2f}"]])
    report.add(f"Fitted equation: `{equation}`")
    return SimpleFit(best, numeric_features, X, y)
