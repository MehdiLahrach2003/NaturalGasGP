import sys
from pathlib import Path

# -------------------------------------------------------------------
# Make sure the project root is on sys.path so that `gp_model` imports
# correctly when this script is executed directly.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gp_model.gas_forward_gp import GasForwardGP, GasForwardGPConfig


def load_synthetic_data(project_root: Path) -> pd.DataFrame:
    """
    Load the synthetic forward curve dataset generated in Step 1.
    """
    csv_path = project_root / "data" / "synthetic" / "synthetic_forward_curve.csv"
    df = pd.read_csv(csv_path)
    return df


def plot_gp_reconstruction(
    df: pd.DataFrame,
    ttm_grid: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """
    Plot:
    - True synthetic curve
    - Observed noisy quotes
    - GP mean reconstruction
    - 95% confidence band
    """
    plt.figure(figsize=(10, 6))

    # True curve
    plt.plot(
        df["ttm_years"],
        df["forward_true"],
        label="True forward curve (synthetic)",
        linewidth=2,
    )

    # Observed quotes
    obs = df[df["is_observed"]]
    plt.scatter(
        obs["ttm_years"],
        obs["forward_observed"],
        label="Observed forwards (noisy & sparse)",
        marker="o",
    )

    # GP mean prediction
    plt.plot(
        ttm_grid,
        mean_pred,
        label="GP mean (RBF kernel)",
        linewidth=2,
        linestyle="--",
    )

    # 95% confidence interval
    lower = mean_pred - 1.96 * std_pred
    upper = mean_pred + 1.96 * std_pred
    plt.fill_between(
        ttm_grid,
        lower,
        upper,
        alpha=0.3,
        label="95% confidence band",
    )

    plt.xlabel("Time to maturity (years)")
    plt.ylabel("Forward price (USD/MMBtu)")
    plt.title("GP Reconstruction of Synthetic Gas Forward Curve (RBF kernel)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)

    plt.show()


def main():
    # 1. Project root (parent of /experiments)
    project_root = PROJECT_ROOT

    # 2. Load synthetic data
    df = load_synthetic_data(project_root)

    # 3. Select observed points only (where we have quotes)
    observed = df[df["is_observed"]].copy()
    ttm_obs = observed["ttm_years"].values
    forward_obs = observed["forward_observed"].values

    # 4. Define prediction grid (dense maturities)
    ttm_grid = np.linspace(df["ttm_years"].min(), df["ttm_years"].max(), 200)

    # 5. Build and fit GP model with RBF kernel
    config = GasForwardGPConfig()
    model = GasForwardGP(config=config)

    print("Fitting GP model with RBF + noise kernel...")
    model.fit(ttm_obs, forward_obs)
    print("Fitted kernel:", model.kernel_)

    # 6. Predict on the grid
    mean_pred, std_pred = model.predict(ttm_grid, return_std=True)

    # 7. Plot and save figure
    reports_dir = project_root / "reports" / "synthetic_results"
    fig_path = reports_dir / "gp_rbf_reconstruction.png"

    plot_gp_reconstruction(
        df=df,
        ttm_grid=ttm_grid,
        mean_pred=mean_pred,
        std_pred=std_pred,
        output_path=fig_path,
    )

    print(f"Figure saved to: {fig_path}")


if __name__ == "__main__":
    main()
