from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gp_model.gas_forward_gp import GasForwardGP, GasForwardGPConfig
from gp_model.metrics import regression_metrics, probabilistic_metrics
from gp_model.io import load_yaml


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def _avg_std(std: np.ndarray) -> float:
    std = np.asarray(std, dtype=float)
    return float(np.mean(std))


def _drop_nan(ttm: np.ndarray, y: np.ndarray):
    mask = np.isfinite(ttm) & np.isfinite(y)
    return ttm[mask], y[mask]


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of columns {candidates} found. Available: {list(df.columns)}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    cfg_path = PROJECT_ROOT / "config" / "default.yaml"
    print(f"[INFO] Loading config: {cfg_path}")
    cfg = load_yaml(cfg_path)

    # -----------------------------------------------------------------
    # Load data (DIRECTLY via pandas → zero magic)
    # -----------------------------------------------------------------
    csv_path = PROJECT_ROOT / cfg["data"]["synthetic_csv"]
    print(f"[INFO] Loading CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    t_col = _pick_column(df, ["ttm_years", "ttm"])
    y_obs_col = _pick_column(df, ["forward_observed", "forward_price"])
    y_true_col = _pick_column(df, ["forward_true", "true_forward"])

    t_obs = df[t_col].values
    y_obs = df[y_obs_col].values

    t_obs, y_obs = _drop_nan(t_obs, y_obs)

    # True curve (interpolated on grid)
    t_true = df[t_col].values
    y_true = df[y_true_col].values
    mask_true = np.isfinite(t_true) & np.isfinite(y_true)
    t_true = t_true[mask_true]
    y_true = y_true[mask_true]

    # -----------------------------------------------------------------
    # Grid
    # -----------------------------------------------------------------
    grid_cfg = cfg["grid"]
    t_grid = np.linspace(
        float(grid_cfg["ttm_min"]),
        float(grid_cfg["ttm_max"]),
        int(grid_cfg["n_points"]),
    )

    y_true_grid = np.interp(t_grid, t_true, y_true)

    # -----------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------
    gp_cfg = cfg["gp"]

    cfg_rbf = GasForwardGPConfig(
        kernel_type="rbf",
        n_restarts_optimizer=int(gp_cfg["n_restarts_optimizer"]),
        normalize_y=bool(gp_cfg["normalize_y"]),
    )

    cfg_seasonal = GasForwardGPConfig(
        kernel_type="rbf_periodic",
        n_restarts_optimizer=int(gp_cfg["n_restarts_optimizer"]),
        normalize_y=bool(gp_cfg["normalize_y"]),
    )

    print("[INFO] Fitting RBF GP...")
    model_rbf = GasForwardGP(cfg_rbf)
    model_rbf.fit(t_obs, y_obs)
    mean_rbf, std_rbf = model_rbf.predict(t_grid)

    print("[INFO] Fitting RBF + Periodic GP...")
    model_seas = GasForwardGP(cfg_seasonal)
    model_seas.fit(t_obs, y_obs)
    mean_seas, std_seas = model_seas.predict(t_grid)

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------
    reg_rbf = regression_metrics(y_true_grid, mean_rbf)
    prob_rbf = probabilistic_metrics(y_true_grid, mean_rbf, std_rbf)

    reg_seas = regression_metrics(y_true_grid, mean_seas)
    prob_seas = probabilistic_metrics(y_true_grid, mean_seas, std_seas)

    lml_rbf = model_rbf.gp.log_marginal_likelihood(model_rbf.gp.kernel_.theta)
    lml_seas = model_seas.gp.log_marginal_likelihood(model_seas.gp.kernel_.theta)

    results = pd.DataFrame(
        [
            {
                "model": "RBF",
                "rmse": reg_rbf.rmse,
                "mae": reg_rbf.mae,
                "mape": reg_rbf.mape,
                "avg_std": _avg_std(std_rbf),
                "coverage_95": prob_rbf.coverage_95,
                "nll_gaussian": prob_rbf.nll_gaussian,
                "log_marginal_likelihood": lml_rbf,
            },
            {
                "model": "RBF + Periodic",
                "rmse": reg_seas.rmse,
                "mae": reg_seas.mae,
                "mape": reg_seas.mape,
                "avg_std": _avg_std(std_seas),
                "coverage_95": prob_seas.coverage_95,
                "nll_gaussian": prob_seas.nll_gaussian,
                "log_marginal_likelihood": lml_seas,
            },
        ]
    ).sort_values("rmse")

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------
    reports_dir = PROJECT_ROOT / cfg["reports"]["synthetic_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_csv = reports_dir / "gp_model_comparison.csv"
    results.to_csv(out_csv, index=False)

    print("\n=== GP model comparison ===")
    print(results)
    print(f"[INFO] CSV saved to {out_csv}")

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(t_grid, y_true_grid, label="True curve")
    axes[0].scatter(t_obs, y_obs, s=40, label="Observed")
    axes[0].plot(t_grid, mean_rbf, label="RBF")
    axes[0].fill_between(
        t_grid, mean_rbf - 1.96 * std_rbf, mean_rbf + 1.96 * std_rbf, alpha=0.2
    )
    axes[0].plot(t_grid, mean_seas, label="RBF + Periodic")
    axes[0].fill_between(
        t_grid, mean_seas - 1.96 * std_seas, mean_seas + 1.96 * std_seas, alpha=0.2
    )
    axes[0].legend()
    axes[0].set_title("GP reconstruction")

    axes[1].bar(results["model"], results["rmse"])
    axes[1].set_title("RMSE")

    axes[2].bar(results["model"], results["avg_std"])
    axes[2].set_title("Average predictive std")

    plt.tight_layout()

    out_png = reports_dir / "gp_model_comparison.png"
    plt.savefig(out_png, dpi=300)
    print(f"[INFO] Figure saved to {out_png}")

    plt.show()


if __name__ == "__main__":
    main()
