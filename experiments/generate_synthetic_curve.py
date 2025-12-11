import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def true_forward_curve(ttm_years: np.ndarray) -> np.ndarray:
    """
    Synthetic 'true' natural gas forward curve used for testing the GP model.

    Parameters
    ----------
    ttm_years : np.ndarray
        Time to maturity in years (1D array).

    Returns
    -------
    np.ndarray
        Theoretical forward curve values at each maturity.
    """
    ttm_years = np.asarray(ttm_years)

    # Global upward trend
    trend = 2.0 + 0.4 * ttm_years

    # Annual seasonality (1-year periodic cycle)
    seasonality = 0.6 * np.sin(2 * np.pi * ttm_years)

    # Small curvature effect (to make the curve slightly more complex)
    curvature = -0.1 * (ttm_years - 1.5) ** 2

    return trend + seasonality + curvature


def generate_synthetic_data(
    n_points_full: int = 36,
    n_points_observed: int = 12,
    ttm_min: float = 0.1,
    ttm_max: float = 3.0,
    noise_std: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic forward curve data for testing Gaussian Processes.

    The output contains:
    - A smooth underlying "true" forward curve.
    - A sparse set of observed maturities (simulating illiquid markets).
    - Random noise added to the observed points.

    Returns
    -------
    pd.DataFrame with columns:
        ttm_years, forward_true, is_observed, forward_observed
    """
    rng = np.random.RandomState(random_state)

    # Full grid of maturities (time to maturity in years)
    ttm_full = np.linspace(ttm_min, ttm_max, n_points_full)
    forward_true = true_forward_curve(ttm_full)

    # Randomly select which maturities are observed (illiquidity simulation)
    idx_observed = np.sort(
        rng.choice(n_points_full, size=n_points_observed, replace=False)
    )
    is_observed = np.zeros(n_points_full, dtype=bool)
    is_observed[idx_observed] = True

    # Observed forward prices = true price + noise
    forward_observed = np.full_like(forward_true, fill_value=np.nan, dtype=float)
    forward_observed[idx_observed] = (
        forward_true[idx_observed] + rng.normal(scale=noise_std, size=n_points_observed)
    )

    df = pd.DataFrame(
        {
            "ttm_years": ttm_full,
            "forward_true": forward_true,
            "is_observed": is_observed,
            "forward_observed": forward_observed,
        }
        
    )

    return df


def plot_synthetic_curve(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """
    Plot the synthetic forward curve: true curve + noisy observed quotes.

    Parameters
    ----------
    df : pd.DataFrame
        Synthetic dataset produced by generate_synthetic_data().
    output_path : Path or None
        If provided, the figure is saved to this path.
    """
    plt.figure(figsize=(10, 6))

    # Plot the underlying true forward curve
    plt.plot(
        df["ttm_years"],
        df["forward_true"],
        label="True forward curve (synthetic)",
        linewidth=2,
    )

    # Plot observed (noisy) quotes
    obs = df[df["is_observed"]]
    plt.scatter(
        obs["ttm_years"],
        obs["forward_observed"],
        label="Observed forwards (noisy & sparse)",
        marker="o",
    )

    plt.xlabel("Time to maturity (years)")
    plt.ylabel("Forward price (USD/MMBtu)")
    plt.title("Synthetic Natural Gas Forward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to file if a path is provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)

    plt.show()


def main():
    """
    Entry point of the script:
    - Generate synthetic forward curve data
    - Save CSV
    - Plot and save figure
    """
    project_root = Path(__file__).resolve().parents[1]

    data_dir = project_root / "data" / "synthetic"
    reports_dir = project_root / "reports" / "synthetic_results"

    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Generate the synthetic dataset
    df = generate_synthetic_data()

    # Save CSV file
    csv_path = data_dir / "synthetic_forward_curve.csv"
    df.to_csv(csv_path, index=False)
    print(f"Synthetic dataset saved to: {csv_path}")

    # Plot and save figure
    fig_path = reports_dir / "synthetic_forward_curve.png"
    plot_synthetic_curve(df, output_path=fig_path)
    print(f"Plot saved to: {fig_path}")


if __name__ == "__main__":
    main()
