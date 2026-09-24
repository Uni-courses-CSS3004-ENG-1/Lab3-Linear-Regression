"""Step 2 — Load and inspect the data, then drop rows with missing power_hp."""
from dataclasses import dataclass

import pandas as pd

from lab.config import DATA_PATH
from lab.report import Report


@dataclass(frozen=True)
class CleanData:
    df: pd.DataFrame          # the rows every model uses
    missing: pd.DataFrame     # the dropped rows (Step 7 describes them)
    rows_before: int
    rows_dropped: int
    share_dropped: str        # e.g. "1.51%"


def run(report: Report) -> CleanData:
    report.section("Step 2 — Load and inspect the data")

    df = pd.read_csv(DATA_PATH)
    print(df.shape)
    print(df.dtypes)
    print(df.describe())
    print(df.isna().sum())

    missing = df[df["power_hp"].isna()]
    print("\nRows with missing power_hp:")
    print(missing[["model", "year", "region"]])

    rows_before = len(df)
    df = df.dropna(subset=["power_hp"])
    rows_dropped = rows_before - len(df)
    share_dropped = f"{rows_dropped / rows_before:.2%}"
    print(f"\nDropped {rows_dropped} of {rows_before} rows with missing power_hp "
          f"({share_dropped}); {len(df)} rows remain.")
    report.table(
        ["Rows loaded", "Rows dropped (missing power_hp)", "Share dropped", "Rows used"],
        [[rows_before, rows_dropped, share_dropped, len(df)]],
    )
    return CleanData(df, missing, rows_before, rows_dropped, share_dropped)
