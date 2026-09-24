"""The three figures saved to figures/ (Steps 3 and 6)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from lab.config import FIG_DIR, TARGET


def plot_simple_fit(X: pd.DataFrame, y: pd.Series, feature: str,
                    model: LinearRegression, equation: str) -> None:
    """Save the scatter plot, then the same plot with the fitted line on top."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X[feature], y, s=14, alpha=0.6)
    ax.set_xlabel(feature)
    ax.set_ylabel(TARGET)
    ax.set_title(f"{TARGET} vs {feature}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step3_scatter.png", dpi=150)

    x_line = pd.DataFrame({feature: np.linspace(X[feature].min(), X[feature].max(), 200)})
    ax.plot(x_line[feature], model.predict(x_line), color="tab:red", linewidth=2, label=equation)
    ax.set_title(f"{TARGET} vs {feature} with the fitted line")
    ax.legend()
    fig.savefig(FIG_DIR / "step3_fit.png", dpi=150)
    plt.close(fig)


def plot_r2_vs_degree(poly: pd.DataFrame, feature: str) -> None:
    """Train and test R² against polynomial degree on the same axes."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(poly.index, poly.train_r2, marker="o", label="Train R² (5-fold mean)")
    ax.plot(poly.index, poly.test_r2, marker="s", label="Test R² (5-fold mean)")
    ax.set_xticks(poly.index)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("R²")
    ax.set_title(f"Polynomial regression on {feature}: R² vs degree")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step6_r2_vs_degree.png", dpi=150)
    plt.close(fig)
