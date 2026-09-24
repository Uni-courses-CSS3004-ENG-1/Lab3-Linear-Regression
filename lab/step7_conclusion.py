"""Step 7 — Conclusion: pick the model to keep and write the 150–250 word answer."""
from lab.config import MIN_GAIN
from lab.report import Report, code_list
from lab.step2_data import CleanData
from lab.step3_simple import SimpleFit
from lab.step4_evaluate import SimpleScores
from lab.step5_multiple import FeatureSearch
from lab.step6_polynomial import PolySearch


def run(report: Report, data: CleanData, fit: SimpleFit, simple: SimpleScores,
        search: FeatureSearch, poly: PolySearch) -> None:
    report.section("Step 7 — Conclusion")

    best = fit.best
    table, top = search.table, search.top
    chosen = table.features[search.best_row]
    chosen_r2 = table.mean_r2[search.best_row]
    chosen_rmse = table.rmse[search.best_row]
    all_r2 = table.mean_r2.iloc[-1]
    excluded = fit.numeric_features[len(chosen):]
    weak_kept = [name for name in search.weak.added if name in chosen]
    year_min, year_max = data.df["year"].min(), data.df["year"].max()

    if max(simple.cv_mean, poly.best_r2) >= chosen_r2:
        print("WARNING: a Step 3/6 model matches or beats the best Step 5 model; "
              "the generated conclusion does not fit these numbers — rewrite it by hand.")
        report.add("> **Warning.** The conclusion below assumes the Step 5 model is best, "
                   "which these numbers do not support. Rewrite it by hand.", "")

    choice = [
        f"I would use the multiple linear regression from Step 5 on {len(chosen)} features: "
        f"{code_list(chosen)}.",
        f"It has the highest mean 5-fold test R² of every model I fitted, {chosen_r2:.4f}, "
        f"against {simple.cv_mean:.4f} for simple regression on `{best}` and "
        f"{poly.best_r2:.4f} for the best polynomial (degree {poly.best_degree}).",
        f"Its cross-validated RMSE is {chosen_rmse:.2f} km/L versus {simple.cv_rmse:.2f} km/L "
        f"for `{best}` alone, so a typical prediction for a new car is off by about "
        f"{chosen_rmse:.1f} km/L.",
    ]
    if top.added in chosen:
        choice.append(f"Adding `{top.added}` was worth {top.change:+.4f} R², while bending the "
                      f"`{best}` line into a curve gained only "
                      f"{poly.best_r2 - simple.cv_mean:+.4f}.")
        if top.added == "year":
            choice.append(f"Cars became more efficient over the {year_min}–{year_max} model "
                          f"years even at the same `{best}`, and no curve in `{best}` alone "
                          "can capture that.")
    if excluded:
        choice.append(f"I left out {code_list(excluded)}: with every feature the mean test R² "
                      f"is {all_r2:.4f}, no better than this model, and each extra input is one "
                      "more thing to measure for a new car.")
    if weak_kept:
        one = len(weak_kept) == 1
        choice.append(f"{code_list(weak_kept)} added less than {MIN_GAIN} R² when "
                      f"{'it' if one else 'they'} entered the model, so a leaner model without "
                      f"{'it' if one else 'them'} is worth testing next.")

    caveats = [
        f"Only {data.rows_dropped} of {data.rows_before} rows ({data.share_dropped}) were "
        f"dropped for missing `power_hp`; they come from {data.missing['year'].nunique()} "
        f"different model years and {data.missing['region'].nunique()} regions, and so few "
        f"rows cannot noticeably move coefficients fitted on the remaining {len(data.df)} cars.",
        f"One caution: the model only saw cars from {year_min}–{year_max}, so "
        + ("its `year` effect should not be extrapolated to modern cars."
           if "year" in chosen else "it should not be trusted for modern cars."),
    ]
    report.paragraph(*choice)
    report.paragraph(*caveats)

    word_count = len(" ".join(choice + caveats).split())
    report.add(f"_Word count: {word_count}_")
    print(f"Chosen model: multiple linear regression on {', '.join(chosen)}")
    print(f"Mean test R² = {chosen_r2:.4f}, CV RMSE = {chosen_rmse:.2f} km/L")
    print(f"Conclusion word count: {word_count}"
          + ("" if 150 <= word_count <= 250 else "  WARNING: the lab asks for 150–250 words"))
