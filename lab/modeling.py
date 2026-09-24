"""Model building and scoring helpers used by Steps 4–7."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from lab.config import CV


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R²": r2_score(y_true, y_pred),
    }


def cv_scores(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> dict[str, np.ndarray]:
    """Per-fold train R², test R² and test RMSE on the shared 5-fold split."""
    scores = cross_validate(model, X, y, cv=CV, return_train_score=True,
                            scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"})
    return {
        "train_r2": scores["train_r2"],
        "test_r2": scores["test_r2"],
        "test_rmse": -scores["test_rmse"],
    }


def poly_model(degree: int) -> Pipeline:
    """Scale first, so high powers of the raw feature don't break least squares."""
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    )
