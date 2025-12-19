# experiments/henry_hub_metrics.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Small YAML loader (robust)
# -----------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Robust YAML loader.
    Tries gp_model.io.load_yaml if available, otherwise uses PyYAML directly.
    """
    try:
        from gp_model.io import load_yaml as _load_yaml  # type: ignore
        return _load_yaml(path)
    except Exception:
        import yaml  # PyYAML
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def project_root() -> Path:
    # experiments/ -> repo root
    return Path(__file__).resolve().parents[1]


def get_cfg(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def pick_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Pick (ttm_col, y_col) with sensible fallbacks.
    """
    ttm_candidates = ["ttm_years", "ttm", "maturity_years", "maturity"]
    y_candidates = ["forward_price", "forward_observed", "price", "y"]

    ttm_col = next((c for c in ttm_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)

    if ttm_col is None:
        raise KeyError(f"Could not find a TTM column in CSV. Tried: {ttm_candidates}. Got: {list(df.columns)}")
    if y_col is None:
        raise KeyError(f"Could not find a price column in CSV. Tried: {y_candidates}. Got: {list(df.columns)}")

    return ttm_col, y_col


def build_model_from_cfg(cfg: Dict[str, Any]):
    """
    Builds GasForwardGP from config, matching your current gp_model.gas_forward_gp API.
    """
    from gp_model.gas_forward_gp import GasForwardGP, GasForwardGPConfig  # type: ignore
    from gp_model.kernels import RBFKernelParams, PeriodicKernelParams, NoiseKernelParams  # type: ignore

    kernel_type = get_cfg(cfg, "gp", "kernel_type", default="rbf_periodic")

    n_restarts = int(get_cfg(cfg, "gp", "n_restarts_optimizer", default=3))
    normalize_y = bool(get_cfg(cfg, "gp", "normalize_y", default=True))
    random_state = int(get_cfg(cfg, "gp", "random_state", default=42))

    # Params from YAML
    rbf_ls = float(get_cfg(cfg, "gp", "rbf", "length_scale", default=0.6))
    per_ls = float(get_cfg(cfg, "gp", "periodic", "length_scale", default=0.8))

    # Your YAML uses gp.periodic.period
    period = get_cfg(cfg, "gp", "periodic", "period", default=None)
    if period is None:
        # tolerate other legacy keys, just in case
        period = get_cfg(cfg, "gp", "periodic", "period_years", default=None)
    if period is None:
        period = get_cfg(cfg, "gp", "periodic", "periodicity", default=1.0)
    period = float(period)

    # Alpha in your YAML used to exist; if missing, we default.
    alpha = get_cfg(cfg, "gp", "alpha", default=None)
    if alpha is None:
        alpha = 1e-4
        print("[WARN] gp.alpha not found, using default 1e-4")
    noise_level = float(alpha)

    # IMPORTANT: PeriodicKernelParams field name in your project is very likely `periodicity`
    # (sklearn ExpSineSquared uses periodicity=...).
    rbf_params = RBFKernelParams(length_scale=rbf_ls)
    periodic_params = PeriodicKernelParams(length_scale=per_ls, periodicity=period)
    noise_params = NoiseKernelParams(noise_level=noise_level)

    model_cfg = GasForwardGPConfig(
        kernel_type=kernel_type,
        n_restarts_optimizer=n_restarts,
        normalize_y=normalize_y,
        random_state=random_state,
        rbf_params=rbf_params,
        periodic_params=periodic_params,
        noise_params=noise_params,
    )
    return GasForwardGP(model_cfg)


def loocv_predictions(model_builder, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leave-One-Out CV:
      For each i, fit on all points except i, predict at x[i].
    Returns (mean_cv, std_cv).
    """
    n = len(x)
    mean_cv = np.empty(n, dtype=float)
    std_cv = np.empty(n, dtype=float)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        model = model_builder()
        model.fit(x[mask], y[mask])

        m, s = model.predict(np.array([x[i]]), return_std=True)
        mean_cv[i] = float(np.asarray(m).ravel()[0])
        std_cv[i] = float(np.asarray(s).ravel()[0])

    return mean_cv, std_cv


def main() -> None:
    ROOT = project_root()
    cfg_path = ROOT / "config" / "default.yaml"

    print(f"[INFO] Loading config: {cfg_path}")
    cfg = load_yaml(cfg_path)

    henry_csv_rel = get_cfg(cfg, "data", "henry_hub_csv", default="data/henry_hub/henry_hub.csv")
    henry_csv = (ROOT / henry_csv_rel).resolve()
    print(f"[INFO] Loading CSV: {henry_csv}")

    if not henry_csv.exists():
        raise FileNotFoundError(
            f"Henry Hub CSV not found: {henry_csv}\n"
            f"Put it here (non commité): data/henry_hub/henry_hub.csv"
        )

    df = pd.read_csv(henry_csv)
    ttm_col, y_col = pick_columns(df)
    print(f"[INFO] Using columns: ttm='{ttm_col}', y='{y_col}'")

    # Clean NaNs (important!)
    df = df[[ttm_col, y_col]].copy()
    df[ttm_col] = pd.to_numeric(df[ttm_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    before = len(df)
    df = df.dropna()
    after = len(df)
    if after < before:
        print(f"[WARN] Dropped {before - after} rows with NaN/invalid values")

    df = df.sort_values(ttm_col)

    x = df[ttm_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    if len(x) < 4:
        print("[WARN] Very few points. LOOCV will be noisy; still runs but interpret with care.")

    # Builder so each fold gets a fresh model
    def _builder():
        return build_model_from_cfg(cfg)

    print("[INFO] Running LOOCV...")
    mean_cv, std_cv = loocv_predictions(_builder, x, y)

    # Metrics (out-of-sample via LOOCV)
    from gp_model.metrics import regression_metrics, probabilistic_metrics  # type: ignore

    reg = regression_metrics(y, mean_cv)
    prob = probabilistic_metrics(y, mean_cv, std_cv)

    print("\n=== Henry Hub — GP Metrics (LOOCV / out-of-sample) ===")
    print(f"RMSE         : {reg.rmse:.6f}")
    print(f"MAE          : {reg.mae:.6f}")
    print(f"MAPE         : {reg.mape}")
    print(f"Coverage 95% : {prob.coverage_95:.3f}")
    print(f"Gaussian NLL : {prob.nll_gaussian:.6f}")
    print("=====================================================\n")

    # Fit once on full data for a smooth curve plot
    model_full = _builder()
    model_full.fit(x, y)

    grid_min = float(get_cfg(cfg, "grid", "ttm_min", default=float(np.min(x))))
    grid_max = float(get_cfg(cfg, "grid", "ttm_max", default=float(np.max(x))))
    n_points = int(get_cfg(cfg, "grid", "n_points", default=250))
    x_grid = np.linspace(grid_min, grid_max, n_points)

    mean_grid, std_grid = model_full.predict(x_grid, return_std=True)
    mean_grid = np.asarray(mean_grid).ravel()
    std_grid = np.asarray(std_grid).ravel()

    # Output dir
    out_dir_rel = get_cfg(cfg, "reports", "henry_hub_dir", default="reports/henry_hub_results")
    out_dir = ROOT / out_dir_rel
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / "henry_hub_loocv_metrics.png"

    # Plot: full fit + LOOCV predictions at the observation points
    plt.figure(figsize=(12, 6))

    # Full fit band
    plt.plot(x_grid, mean_grid, linestyle="--", label="GP mean (fit on all data)")
    plt.fill_between(
        x_grid,
        mean_grid - 1.96 * std_grid,
        mean_grid + 1.96 * std_grid,
        alpha=0.2,
        label="95% confidence band (fit on all data)",
    )

    # Observed points
    plt.scatter(x, y, s=45, label="Observed (Henry Hub)")

    # LOOCV preds with error bars
    plt.errorbar(
        x,
        mean_cv,
        yerr=1.96 * std_cv,
        fmt="o",
        capsize=3,
        label="LOOCV predictions (mean ± 1.96 std)",
    )

    plt.title("Henry Hub — GP Reconstruction + LOOCV diagnostics")
    plt.xlabel("Time to maturity (years)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=300)
    print(f"[INFO] Saved figure: {out_png}")
    print(f"[INFO] Fitted kernel:\n{model_full.kernel_}")

    plt.show()


if __name__ == "__main__":
    main()
