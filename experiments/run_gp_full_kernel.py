from __future__ import annotations

from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from gp_model.gas_forward_gp import GasForwardGP, GasForwardGPConfig
from gp_model.kernels import (
    RBFKernelParams,
    PeriodicKernelParams,
    NoiseKernelParams,
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Return the first column name found in df among candidates.
    Raise a clear error if none exist.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        "None of the expected columns were found.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print(f"[INFO] Loading config: {CONFIG_PATH}")
    cfg = load_yaml(CONFIG_PATH)

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------

    synthetic_csv = PROJECT_ROOT / cfg["data"]["synthetic_csv"]
    print(f"[INFO] Loading CSV: {synthetic_csv}")
    df = pd.read_csv(synthetic_csv)

    # Always required
    ttm_col = pick_column(df, ["ttm_years", "ttm", "maturity_years", "T"])
    ttm = df[ttm_col].values

    # Observed target (try common names used across your scripts)
    y_col = pick_column(
        df,
        [
            "forward_price",       # (your current script expected this)
            "forward_observed",    # (common in your synthetic generator)
            "forward",             # generic
            "price",               # generic
            "y",                   # generic
            "forward_obs",         # alternate
        ],
    )
    y_obs = df[y_col].values

    print(f"[INFO] Using columns: ttm='{ttm_col}', y='{y_col}'")

    # Grid for prediction
    grid_cfg = cfg["grid"]
    ttm_grid = np.linspace(
        float(grid_cfg["ttm_min"]),
        float(grid_cfg["ttm_max"]),
        int(grid_cfg["n_points"]),
    )

    # -----------------------------------------------------------------
    # GP config (Full kernel = RBF + periodic + noise)
    # -----------------------------------------------------------------

    gp_cfg = cfg.get("gp", {})

    # noise_level stored under gp.alpha (your YAML) or gp.noise_level (fallback)
    if "alpha" in gp_cfg:
        noise_level = float(gp_cfg["alpha"])
    elif "noise_level" in gp_cfg:
        noise_level = float(gp_cfg["noise_level"])
    else:
        noise_level = 1.0e-4
        print("[WARN] gp.alpha not found, using default 1e-4")

    model_config = GasForwardGPConfig(
        kernel_type="rbf_periodic",
        normalize_y=bool(gp_cfg.get("normalize_y", True)),
        n_restarts_optimizer=int(gp_cfg.get("n_restarts_optimizer", 3)),
        rbf_params=RBFKernelParams(
            length_scale=float(gp_cfg["rbf"]["length_scale"])
        ),
        periodic_params=PeriodicKernelParams(
            length_scale=float(gp_cfg["periodic"]["length_scale"]),
            period_years=float(gp_cfg["periodic"]["period_years"]),
        ),
        noise_params=NoiseKernelParams(
            noise_level=noise_level
        ),
    )

    # -----------------------------------------------------------------
    # Fit + predict
    # -----------------------------------------------------------------

    model = GasForwardGP(model_config)
    model.fit(ttm, y_obs)

    mean_pred, std_pred = model.predict(ttm_grid, return_std=True)

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------

    out_dir = PROJECT_ROOT / cfg["reports"]["synthetic_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gp_full_kernel_reconstruction.png"

    plt.figure(figsize=(10, 5))

    plt.plot(ttm, y_obs, "o", label="Observed forwards")
    plt.plot(ttm_grid, mean_pred, "--", label="GP mean (RBF + Periodic)")

    plt.fill_between(
        ttm_grid,
        mean_pred - 1.96 * std_pred,
        mean_pred + 1.96 * std_pred,
        alpha=0.25,
        label="95% confidence interval",
    )

    plt.xlabel("Time to maturity (years)")
    plt.ylabel("Forward price (USD/MMBtu)")
    plt.title("GP Reconstruction – Full kernel (RBF + Periodic)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Figure saved to: {out_path}")
    print(f"[INFO] Fitted kernel: {model.kernel_}")

    plt.show()


if __name__ == "__main__":
    main()
