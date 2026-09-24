"""Step 5 — Multiple linear regression: add features one at a time, best-correlated first."""
from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LinearRegression

from lab.config import MIN_GAIN
from lab.modeling import cv_scores
from lab.report import Report, code_list
from lab.step3_simple import SimpleFit


@dataclass(frozen=True)
class FeatureSearch:
    table: pd.DataFrame   # one row per model: features, added, mean_r2, std_r2, rmse, change
    weak: pd.DataFrame    # additions that changed mean test R² by less than MIN_GAIN
    top: pd.Series        # the addition with the biggest gain
    best_row: int         # row of the model with the highest mean test R²


def run(report: Report, df: pd.DataFrame, fit: SimpleFit) -> FeatureSearch:
    report.section("Step 5 — Multiple linear regression")

    # One row per model: the k best-correlated features, for k = 1, 2, ...
    rows = []
    for k in range(1, len(fit.numeric_features) + 1):
        features = fit.numeric_features[:k]
        scores = cv_scores(LinearRegression(), df[features], fit.y)
        rows.append({
            "features": features,
            "added": features[-1],
            "mean_r2": scores["test_r2"].mean(),
            "std_r2": scores["test_r2"].std(),
            "rmse": scores["test_rmse"].mean(),
        })
    table = pd.DataFrame(rows)
    table["change"] = table["mean_r2"].diff()  # NaN for the first row

    md_rows = []
    for i, row in table.iterrows():
        label = row.added if i == 0 else "+ " + row.added
        change_text = "—" if i == 0 else f"{row.change:+.4f}"
        md_rows.append([label, f"{row.mean_r2:.4f}", f"{row.std_r2:.4f}", change_text])
        print(f"{label:<22} mean test R² = {row.mean_r2:.4f} "
              f"(std {row.std_r2:.4f})  change = {change_text}")
    report.table(["Features used (cumulative)", "Mean test R² (5-fold)", "Std",
                  "Change vs previous row"], md_rows)

    additions = table.iloc[1:]
    search = FeatureSearch(
        table=table,
        weak=additions[additions.change < MIN_GAIN],
        top=additions.loc[additions.change.idxmax()],
        best_row=table.mean_r2.idxmax(),
    )
    report.paragraph(*_answer(search, df, fit.numeric_features))
    return search


def _answer(search: FeatureSearch, df: pd.DataFrame, numeric_features: list[str]) -> list[str]:
    """Did test R² improve every time a feature was added?"""
    table, weak, top = search.table, search.weak, search.top
    n_additions = len(table) - 1
    feature_corr = df[numeric_features].corr()

    answer = ["**Did test R² improve every time a feature was added?**"]
    if len(weak):
        answer.append(
            f"No. {len(weak)} of the {n_additions} additions moved the mean test R² by less "
            f"than {MIN_GAIN}: "
            + ", ".join(f"`{r.added}` ({r.change:+.4f})" for r in weak.itertuples()) + "."
        )
        lowered = list(weak.added[weak.change < 0])
        if lowered:
            answer.append(f"Adding {code_list(lowered)} actually lowered it.")
        # Explain each weak feature by the earlier feature it overlaps with most.
        for name in weak.added:
            earlier = numeric_features[:numeric_features.index(name)]
            partner = feature_corr.loc[name, earlier].abs().idxmax()
            answer.append(f"`{name}` overlaps with `{partner}` "
                          f"(r = {feature_corr.loc[name, partner]:+.2f}), "
                          "which was already in the model.")
    else:
        answer.append(f"Yes. Every addition raised the mean test R² by at least {MIN_GAIN}.")

    top_rank = numeric_features.index(top.added) + 1
    top_sentence = f"The biggest gain came from `{top.added}` ({top.change:+.4f})"
    if top_rank > 2:
        top_sentence += (f", even though it ranks only #{top_rank} of {len(numeric_features)} "
                         "by correlation with the target")
    if top.added == "year":
        top_sentence += ("; it adds information (how new the car is) that the size and power "
                         "columns do not contain")
    answer.append(top_sentence + ".")

    if search.best_row < len(table) - 1:
        answer.append(f"The model with every numeric feature scores "
                      f"{table.mean_r2.iloc[-1]:.4f}, "
                      f"below the best row ({table.mean_r2[search.best_row]:.4f}).")
    answer.append(
        "So throwing every column into the model is not a free improvement: what a feature adds "
        "depends on what the model already contains, not on its own correlation with the target. "
        "A feature that repeats known information only gives least squares another coefficient "
        "to estimate from the same cars, so each addition has to earn its place on held-out data."
    )
    return answer
