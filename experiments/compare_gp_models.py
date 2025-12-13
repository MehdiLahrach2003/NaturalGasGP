import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gp_model.gas_forward_gp import GasForwardGP, GasForwardGPConfig


# ---------------------------------------------------------------------
# Utility metrics
# ---------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ---------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------
def main():

    # -----------------------------------------------------------------
    # Load synthetic dataset
    # -----------------------------------------------------------------
    data_path = PROJECT_ROOT / "data" / "synthetic" / "synthetic_forward_curve.csv"
    df = pd.read_csv(data_path)

    ttm = df["ttm_years"].values
    forward_true = df["forward_true"].values

    mask_obs = df["is_observed"].values.astype(bool)
    ttm_obs = ttm[mask_obs]
    forward_obs = df["forward_observed"].values[mask_obs]

    ttm_grid = ttm.reshape(-1, 1)
    ttm_obs = ttm_obs.reshape(-1, 1)

    # -----------------------------------------------------------------
    # Define models to compare
    # -----------------------------------------------------------------
    models = {
        "RBF": GasForwardGP(
            GasForwardGPConfig(kernel_type="rbf")
        ),
        "RBF + Periodic": GasForwardGP(
            GasForwardGPConfig(kernel_type="rbf_periodic")
        ),
    }

    results = {}

    # -----------------------------------------------------------------
    # Fit & predict
    # -----------------------------------------------------------------
    for name, model in models.items():
        print(f"\nFitting model: {name}")
        model.fit(ttm_obs, forward_obs)

        mean_pred, std_pred = model.predict(ttm_grid, return_std=True)

        results[name] = {
            "mean": mean_pred,
            "std": std_pred,
            "rmse": rmse(forward_true, mean_pred),
            "avg_std": np.mean(std_pred),
        }

        print(f"RMSE = {results[name]['rmse']:.4f}")
        print(f"Average predictive std = {results[name]['avg_std']:.4f}")

    # -----------------------------------------------------------------
    # Plot comparison
    # -----------------------------------------------------------------
    reports_dir = PROJECT_ROOT / "reports" / "synthetic_results"
    reports_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ---- Curve reconstruction
    ax = axes[0]
    ax.plot(ttm, forward_true, label="True curve", lw=2)
    ax.scatter(ttm_obs, forward_obs, color="black", zorder=3, label="Observed")

    for name, res in results.items():
        ax.plot(ttm, res["mean"], label=name)
        ax.fill_between(
            ttm,
            res["mean"] - 1.96 * res["std"],
            res["mean"] + 1.96 * res["std"],
            alpha=0.2,
        )

    ax.set_title("GP reconstruction")
    ax.set_xlabel("Time to maturity (years)")
    ax.set_ylabel("Forward price (USD/MMBtu)")
    ax.legend()
    ax.grid(True)

    # ---- RMSE comparison
    ax = axes[1]
    ax.bar(results.keys(), [res["rmse"] for res in results.values()])
    ax.set_title("RMSE (vs true curve)")
    ax.grid(True, axis="y")

    # ---- Uncertainty comparison
    ax = axes[2]
    ax.bar(results.keys(), [res["avg_std"] for res in results.values()])
    ax.set_title("Average predictive uncertainty")
    ax.grid(True, axis="y")

    plt.suptitle("Gaussian Process Model Comparison")
    plt.tight_layout()
    plt.savefig(reports_dir / "gp_model_comparison.png", dpi=300)
    plt.show()

    print("\nComparison figure saved to:")
    print(reports_dir / "gp_model_comparison.png")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
