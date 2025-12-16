from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# Make project root importable (so "import gp_model" works when running this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gp_model.gas_forward_gp import GasForwardGP, GasForwardGPConfig
from gp_model.kernels import RBFKernelParams, PeriodicKernelParams, NoiseKernelParams


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pick_first_existing(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    x_candidates = ["ttm_years", "ttm", "maturity_years", "T"]
    y_candidates = ["forward_observed", "forward_price", "y", "price"]

    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)

    if x_col is None or y_col is None:
        raise ValueError(
            f"Colonnes attendues non trouvées. Colonnes disponibles: {list(df.columns)}. "
            f"J'attendais une colonne X parmi {x_candidates} et Y parmi {y_candidates}."
        )
    return x_col, y_col


def main() -> None:
    # ---------- Config ----------
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    print(f"[INFO] Loading config: {config_path}")
    cfg = _load_yaml(config_path)

    data_cfg = cfg.get("data", {})
    reports_cfg = cfg.get("reports", {})
    gp_cfg = cfg.get("gp", {})
    grid_cfg = cfg.get("grid", {})

    synthetic_csv_rel = data_cfg.get("synthetic_csv", "data/synthetic/synthetic_forward_curve.csv")
    synthetic_csv = PROJECT_ROOT / synthetic_csv_rel
    print(f"[INFO] Loading CSV: {synthetic_csv}")

    df = pd.read_csv(synthetic_csv)
    x_col, y_col = _detect_columns(df)
    print(f"[INFO] Using columns: ttm='{x_col}', y='{y_col}'")

    # ---------- Clean NaNs ----------
    before = len(df)
    df = df[[x_col, y_col]].copy()
    df = df.dropna(subset=[x_col, y_col])
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows due to NaNs in '{x_col}' or '{y_col}'")

    if after < 3:
        raise ValueError(f"Not enough data after NaN filtering (n={after}).")

    ttm_obs = df[x_col].to_numpy(dtype=float)
    y_obs = df[y_col].to_numpy(dtype=float)

    # ---------- Grid ----------
    t_min = float(grid_cfg.get("ttm_min", float(np.min(ttm_obs))))
    t_max = float(grid_cfg.get("ttm_max", float(np.max(ttm_obs))))
    n_pts = int(grid_cfg.get("n_points", 250))
    ttm_grid = np.linspace(t_min, t_max, n_pts)

    # ---------- GP params from YAML ----------
    # Our wrapper uses WhiteKernel(noise_level=...), so we map YAML -> noise_level
    # Accept gp.alpha OR gp.noise_level (either is fine).
    noise_level = gp_cfg.get("noise_level", None)
    if noise_level is None:
        noise_level = gp_cfg.get("alpha", None)

    if noise_level is None:
        noise_level = 1e-4
        print("[WARN] gp.noise_level/gp.alpha not found, using default 1e-4")
    noise_level = float(noise_level)

    rbf_block = gp_cfg.get("rbf", {})
    periodic_block = gp_cfg.get("periodic", {})

    rbf_length_scale = float(_pick_first_existing(rbf_block, ["length_scale"], default=0.6))

    # accept "periodicity" or legacy names
    periodic_length_scale = float(_pick_first_existing(periodic_block, ["length_scale"], default=0.8))
    periodicity = float(
        _pick_first_existing(periodic_block, ["periodicity", "period_years", "period"], default=1.0)
    )

    normalize_y = bool(gp_cfg.get("normalize_y", True))
    n_restarts = int(gp_cfg.get("n_restarts_optimizer", 3))

    rbf_params = RBFKernelParams(length_scale=rbf_length_scale)
    periodic_params = PeriodicKernelParams(length_scale=periodic_length_scale, periodicity=periodicity)
    noise_params = NoiseKernelParams(noise_level=noise_level)

    # ---------- Fit model (full kernel) ----------
    model_cfg = GasForwardGPConfig(
        kernel_type="rbf_periodic",
        n_restarts_optimizer=n_restarts,
        normalize_y=normalize_y,
        rbf_params=rbf_params,
        periodic_params=periodic_params,
        noise_params=noise_params,
    )

    model = GasForwardGP(config=model_cfg)
    print("[INFO] Fitting GP (RBF + periodic + noise)...")
    model.fit(ttm_obs, y_obs)

    mean_pred, std_pred = model.predict(ttm_grid, return_std=True)

    # ---------- Plot ----------
    out_dir = PROJECT_ROOT / reports_cfg.get("synthetic_dir", "reports/synthetic_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gp_full_reconstruction.png"

    ci = 1.96
    plt.figure(figsize=(12, 6))
    plt.plot(ttm_obs, y_obs, "o", label="Observed forwards")
    plt.plot(ttm_grid, mean_pred, "--", label="GP mean (RBF + Periodic)")
    plt.fill_between(
        ttm_grid,
        mean_pred - ci * std_pred,
        mean_pred + ci * std_pred,
        alpha=0.2,
        label="95% confidence band",
    )
    plt.xlabel("Time to maturity (years)")
    plt.ylabel("Forward price (USD/MMBtu)")
    plt.title("GP Reconstruction (RBF + Periodic kernel)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Figure saved to: {out_path}")
    print(f"[INFO] Fitted kernel: {model.kernel_}")

    plt.show()


if __name__ == "__main__":
    main()
