"""Settings shared by every step: the seed, file paths and the cross-validation split."""
from pathlib import Path

from sklearn.model_selection import KFold

# Step 4: last 4 digits of my student ID.
# PLACEHOLDER — replace 1234 with the real digits and re-run the script.
RANDOM_STATE = 1234
PLACEHOLDER_STATE = 1234

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cars_fuel_efficiency.csv"
FIG_DIR = ROOT / "figures"
RESULTS_PATH = ROOT / "results.md"

TARGET = "fuel_efficiency_km_per_l"
DEGREES = range(1, 6)

# In the written answers, a change in mean test R² smaller than this counts as
# "no real improvement" (Steps 5–7).
MIN_GAIN = 0.005

# The same shuffled 5-fold split is reused in Steps 4, 5 and 6 so their R² values
# are directly comparable. Shuffling matters: the CSV is ordered by model year.
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
