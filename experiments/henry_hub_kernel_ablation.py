from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel

from gp_model.gas_forward_gp import GasForwardGP, GasForwardGPConfig
from gp_model.metrics import regression_metrics, probabilistic_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


# -------------------------
# Helpers
# -------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_nested(cfg: Dict[str, Any], keys: Tuple[str, ...], default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def set_if_hasattr(obj: Any, attr: str, value: Any) -> Any:
    if hasattr(obj, attr):
        try:
            return replace(obj, **{attr: value})
        except Exception:
            setattr(obj, attr, value)
            return obj
    return obj


def parse_periodic_cfg(periodic_cfg: Dict[str, Any]) -> float:
    for key in ("period", "period_years", "periodicity"):
        if key in periodic_cfg:
            return float(periodic_cfg[key])
    return 1.0


def clean_xy(ttm: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ttm = np.asarray(ttm, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(ttm) & np.isfinite(y)
    return ttm[mask], y[mask]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def loocv_predictions_rbf_family(
    ttm: np.ndarray,
    y: np.ndarray,
    model_cfg: GasForwardGPConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """LOOCV for GasForwardGP (rbf or rbf_periodic)."""
    n = len(ttm)
    means = np.zeros(n, dtype=float)
    stds = np.zeros(n, dtype=float)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        m = GasForwardGP(config=model_cfg)
        m.fit(ttm[mask], y[mask])
        mean_i, std_i = m.predict(np.array([ttm[i]]), return_std=True)
        means[i] = float(mean_i[0])
        stds[i] = float(std_i[0])

    return means, stds


def loocv_predictions_periodic_only(
    ttm: np.ndarray,
    y: np.ndarray,
    length_scale: float,
    period: float,
    noise_level: float = 1e-5,
    n_restarts: int = 3,
    normalize_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """LOOCV for Periodic-only GP: ExpSineSquared + WhiteKernel."""
    n = len(ttm)
    means = np.zeros(n, dtype=float)
    stds = np.zeros(n, dtype=float)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        kernel = ExpSineSquared(length_scale=length_scale, periodicity=period) + WhiteKernel(noise_level=noise_level)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=n_restarts,
            normalize_y=normalize_y,
            random_state=42,
        )
        X_train = np.asarray(ttm[mask]).reshape(-1, 1)
        y_train = np.asarray(y[mask])
        gp.fit(X_train, y_train)

        X_test = np.asarray([ttm[i]]).reshape(-1, 1)
        mean_i, std_i = gp.predict(X_test, return_std=True)
        means[i] = float(mean_i[0])
        stds[i] = float(std_i[0])

    return means, stds


# -------------------------
# Main
# -------------------------
def main() -> None:
    cfg_path = DEFAULT_CONFIG
    cfg = load_yaml(cfg_path)
    print(f"[INFO] Loading config: {cfg_path}")

    csv_rel = get_nested(cfg, ("data", "henry_hub_csv"), default="data/henry_hub/henry_hub.csv")
    csv_path = PROJECT_ROOT / csv_rel
    if not csv_path.exists():
        raise FileNotFoundError(f"Henry Hub CSV not found: {csv_path}")

    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    ttm_col = next((c for c in ["ttm_years", "ttm", "maturity_years"] if c in df.columns), None)
    y_col = next((c for c in ["forward_price", "forward_observed", "price", "y"] if c in df.columns), None)
    if ttm_col is None or y_col is None:
        raise KeyError(f"Need columns like 'ttm_years' and 'forward_price'. Found: {list(df.columns)}")

    print(f"[INFO] Using columns: ttm='{ttm_col}', y='{y_col}'")
    ttm, y = clean_xy(df[ttm_col].values, df[y_col].values)
    n = len(ttm)
    if n < 5:
        raise ValueError(f"Not enough points (n={n})")

    # Grid (for reconstruction plot)
    ttm_min = float(get_nested(cfg, ("grid", "ttm_min"), default=float(np.min(ttm))))
    ttm_max = float(get_nested(cfg, ("grid", "ttm_max"), default=float(np.max(ttm))))
    n_points = int(get_nested(cfg, ("grid", "n_points"), default=250))
    ttm_grid = np.linspace(ttm_min, ttm_max, n_points)

    # Base params from YAML
    gp_cfg = get_nested(cfg, ("gp",), default={}) or {}
    rbf_cfg = gp_cfg.get("rbf", {}) or {}
    per_cfg = gp_cfg.get("periodic", {}) or {}

    rbf_ls = float(rbf_cfg.get("length_scale", 0.6))
    per_ls = float(per_cfg.get("length_scale", 0.8))
    per_period = float(parse_periodic_cfg(per_cfg))

    n_restarts = int(gp_cfg.get("n_restarts_optimizer", 3))
    normalize_y = bool(gp_cfg.get("normalize_y", True))

    # Model configs
    cfg_rbf = GasForwardGPConfig(kernel_type="rbf", n_restarts_optimizer=n_restarts, normalize_y=normalize_y)
    cfg_rbf.rbf_params = set_if_hasattr(cfg_rbf.rbf_params, "length_scale", rbf_ls)

    cfg_rbf_per = GasForwardGPConfig(kernel_type="rbf_periodic", n_restarts_optimizer=n_restarts, normalize_y=normalize_y)
    cfg_rbf_per.rbf_params = set_if_hasattr(cfg_rbf_per.rbf_params, "length_scale", rbf_ls)
    cfg_rbf_per.periodic_params = set_if_hasattr(cfg_rbf_per.periodic_params, "length_scale", per_ls)
    for attr in ("period_years", "period", "periodicity"):
        cfg_rbf_per.periodic_params = set_if_hasattr(cfg_rbf_per.periodic_params, attr, per_period)

    # Fit full models for reconstruction (all data)
    m_rbf = GasForwardGP(config=cfg_rbf)
    m_rbf.fit(ttm, y)
    mean_rbf, std_rbf = m_rbf.predict(ttm_grid, return_std=True)

    m_rbf_per = GasForwardGP(config=cfg_rbf_per)
    m_rbf_per.fit(ttm, y)
    mean_rbf_per, std_rbf_per = m_rbf_per.predict(ttm_grid, return_std=True)

    # Periodic-only model (all data) for reconstruction
    kernel_po = ExpSineSquared(length_scale=per_ls, periodicity=per_period) + WhiteKernel(noise_level=1e-5)
    gp_po = GaussianProcessRegressor(
        kernel=kernel_po,
        alpha=0.0,
        n_restarts_optimizer=n_restarts,
        normalize_y=normalize_y,
        random_state=42,
    )
    gp_po.fit(np.asarray(ttm).reshape(-1, 1), np.asarray(y))
    mean_po, std_po = gp_po.predict(np.asarray(ttm_grid).reshape(-1, 1), return_std=True)

    # LOOCV diagnostics (this is the real “ablation” metric)
    print("[INFO] Running LOOCV for: RBF ...")
    loocv_mean_rbf, loocv_std_rbf = loocv_predictions_rbf_family(ttm, y, cfg_rbf)

    print("[INFO] Running LOOCV for: RBF + Periodic ...")
    loocv_mean_rbf_per, loocv_std_rbf_per = loocv_predictions_rbf_family(ttm, y, cfg_rbf_per)

    print("[INFO] Running LOOCV for: Periodic-only ...")
    loocv_mean_po, loocv_std_po = loocv_predictions_periodic_only(
        ttm, y,
        length_scale=per_ls,
        period=per_period,
        noise_level=1e-5,
        n_restarts=n_restarts,
        normalize_y=normalize_y,
    )

    # Metrics (LOOCV)
    reg_rbf = regression_metrics(y, loocv_mean_rbf)
    reg_rbf_per = regression_metrics(y, loocv_mean_rbf_per)
    reg_po = regression_metrics(y, loocv_mean_po)

    prob_rbf = probabilistic_metrics(y, loocv_mean_rbf, loocv_std_rbf)
    prob_rbf_per = probabilistic_metrics(y, loocv_mean_rbf_per, loocv_std_rbf_per)
    prob_po = probabilistic_metrics(y, loocv_mean_po, loocv_std_po)

    results = pd.DataFrame(
        [
            {
                "model": "RBF",
                "rmse_loocv": reg_rbf.rmse,
                "mae_loocv": reg_rbf.mae,
                "coverage95_loocv": prob_rbf.coverage_95,
                "nll_loocv": prob_rbf.nll_gaussian,
                "avg_std_loocv": float(np.mean(loocv_std_rbf)),
            },
            {
                "model": "RBF + Periodic",
                "rmse_loocv": reg_rbf_per.rmse,
                "mae_loocv": reg_rbf_per.mae,
                "coverage95_loocv": prob_rbf_per.coverage_95,
                "nll_loocv": prob_rbf_per.nll_gaussian,
                "avg_std_loocv": float(np.mean(loocv_std_rbf_per)),
            },
            {
                "model": "Periodic-only",
                "rmse_loocv": reg_po.rmse,
                "mae_loocv": reg_po.mae,
                "coverage95_loocv": prob_po.coverage_95,
                "nll_loocv": prob_po.nll_gaussian,
                "avg_std_loocv": float(np.mean(loocv_std_po)),
            },
        ]
    )

    print("\n=== Henry Hub — Kernel ablation (LOOCV) ===")
    print(results.to_string(index=False))
    print("=========================================\n")

    # Outputs
    reports_dir_rel = get_nested(cfg, ("reports", "henry_hub_dir"), default="reports/henry_hub_results")
    reports_dir = PROJECT_ROOT / reports_dir_rel
    ensure_dir(reports_dir)

    out_png = reports_dir / "henry_hub_kernel_ablation.png"
    out_csv = reports_dir / "henry_hub_kernel_ablation_loocv_metrics.csv"

    results.to_csv(out_csv, index=False)
    print(f"[INFO] Saved metrics: {out_csv}")

    # Plot (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: reconstruction + CI
    ax = axes[0]
    ax.fill_between(ttm_grid, mean_rbf - 1.96 * std_rbf, mean_rbf + 1.96 * std_rbf, alpha=0.15)
    ax.plot(ttm_grid, mean_rbf, label="RBF")

    ax.fill_between(ttm_grid, mean_rbf_per - 1.96 * std_rbf_per, mean_rbf_per + 1.96 * std_rbf_per, alpha=0.15)
    ax.plot(ttm_grid, mean_rbf_per, label="RBF + Periodic")

    ax.fill_between(ttm_grid, mean_po - 1.96 * std_po, mean_po + 1.96 * std_po, alpha=0.15)
    ax.plot(ttm_grid, mean_po, label="Periodic-only")

    ax.scatter(ttm, y, s=50, label="Observed")
    ax.set_title("Reconstruction (fit on all data)")
    ax.set_xlabel("TTM (years)")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()

    # Middle: LOOCV RMSE
    ax = axes[1]
    ax.bar(results["model"], results["rmse_loocv"])
    ax.set_title("LOOCV RMSE")
    ax.grid(True, axis="y")

    # Right: avg predictive std (LOOCV)
    ax = axes[2]
    ax.bar(results["model"], results["avg_std_loocv"])
    ax.set_title("Avg predictive std (LOOCV)")
    ax.grid(True, axis="y")

    plt.suptitle("Henry Hub — Kernel ablation (RBF vs RBF+Periodic vs Periodic-only)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[INFO] Saved figure: {out_png}")
    plt.show()

    print(f"[INFO] Fitted kernel (RBF):\n{m_rbf.kernel_}")
    print(f"[INFO] Fitted kernel (RBF + Periodic):\n{m_rbf_per.kernel_}")
    print(f"[INFO] Fitted kernel (Periodic-only):\n{gp_po.kernel_}")


if __name__ == "__main__":
    main()
