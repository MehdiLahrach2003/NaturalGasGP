import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gp_model.gas_forward_gp import GasForwardGP, GasForwardGPConfig


def load_synthetic(project_root: Path) -> pd.DataFrame:
    csv_path = project_root / "data" / "synthetic" / "synthetic_forward_curve.csv"
    return pd.read_csv(csv_path)


def main():
    project_root = PROJECT_ROOT
    df = load_synthetic(project_root)

    # Observed points only
    obs = df[df["is_observed"]].copy()
    x_obs = obs["ttm_years"].values
    y_obs = obs["forward_observed"].values

    # Prediction grid
    x_grid = np.linspace(df["ttm_years"].min(), df["ttm_years"].max(), 300)

    # -----------------------
    # Model A: RBF (baseline)
    # -----------------------
    model_rbf = GasForwardGP(GasForwardGPConfig(kernel_type="rbf"))
    print("Fitting baseline GP (RBF + noise)...")
    model_rbf.fit(x_obs, y_obs)
    print("Fitted kernel (RBF):", model_rbf.kernel_)

    mean_rbf, std_rbf = model_rbf.predict(x_grid, return_std=True)

    # -----------------------------------
    # Model B: RBF + Periodic (seasonality)
    # -----------------------------------
    model_seasonal = GasForwardGP(GasForwardGPConfig(kernel_type="rbf_periodic"))
    print("\nFitting seasonal GP (RBF + periodic + noise)...")
    model_seasonal.fit(x_obs, y_obs)
    print("Fitted kernel (Seasonal):", model_seasonal.kernel_)

    mean_seas, std_seas = model_seasonal.predict(x_grid, return_std=True)

    # -----------------------
    # Plot comparison
    # -----------------------
    plt.figure(figsize=(11, 6))

    # True curve
    plt.plot(df["ttm_years"], df["forward_true"], linewidth=2, label="True forward curve (synthetic)")

    # Observations
    plt.scatter(x_obs, y_obs, marker="o", label="Observed forwards (noisy & sparse)")

    # Baseline RBF
    plt.plot(x_grid, mean_rbf, linestyle="--", linewidth=2, label="GP mean (RBF)")
    plt.fill_between(
        x_grid,
        mean_rbf - 1.96 * std_rbf,
        mean_rbf + 1.96 * std_rbf,
        alpha=0.20,
        label="95% CI (RBF)",
    )

    # Seasonal
    plt.plot(x_grid, mean_seas, linestyle="-.", linewidth=2, label="GP mean (RBF + Periodic)")
    plt.fill_between(
        x_grid,
        mean_seas - 1.96 * std_seas,
        mean_seas + 1.96 * std_seas,
        alpha=0.20,
        label="95% CI (Seasonal)",
    )

    plt.xlabel("Time to maturity (years)")
    plt.ylabel("Forward price (USD/MMBtu)")
    plt.title("Step 3 — GP Seasonality: RBF vs RBF + Periodic")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save
    out_path = project_root / "reports" / "synthetic_results" / "step3_gp_seasonality_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"\nFigure saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
