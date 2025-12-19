from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

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
    # returns possibly-updated dataclass or object
    if hasattr(obj, attr):
        try:
            # dataclass frozen? use replace
            return replace(obj, **{attr: value})
        except Exception:
            setattr(obj, attr, value)
            return obj
    return obj


def parse_periodic_cfg(periodic_cfg: Dict[str, Any]) -> float:
    # support period OR period_years OR periodicity
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
        raise FileNotFoundError(
            f"Henry Hub CSV not found: {csv_path}\n"
            f"Put your csv at: {csv_rel}"
        )

    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Column detection (robust)
    ttm_col_candidates = ["ttm_years", "ttm", "maturity_years"]
    y_col_candidates = ["forward_price", "forward_observed", "price", "y"]

    ttm_col = next((c for c in ttm_col_candidates if c in df.columns), None)
    y_col = next((c for c in y_col_candidates if c in df.columns), None)

    if ttm_col is None or y_col is None:
        raise KeyError(
            f"Missing required columns. Need one of {ttm_col_candidates} AND one of {y_col_candidates}.\n"
            f"Found columns: {list(df.columns)}"
        )

    print(f"[INFO] Using columns: ttm='{ttm_col}', y='{y_col}'")

    ttm_all, y_all = clean_xy(df[ttm_col].values, df[y_col].values)
    if len(ttm_all) < 5:
        raise ValueError(f"Not enough finite rows after cleaning. Got n={len(ttm_all)}")

    # Split threshold (years)
    split_years = float(get_nested(cfg, ("henry_hub", "split_years"), default=2.0))
    print(f"[INFO] Temporal split at T={split_years:.2f}y (train <= split, test > split)")

    train_mask = ttm_all <= split_years
    test_mask = ~train_mask

    ttm_train, y_train = ttm_all[train_mask], y_all[train_mask]
    ttm_test, y_test = ttm_all[test_mask], y_all[test_mask]

    if len(ttm_test) == 0:
        raise ValueError(
            "No test points found (all maturities <= split). "
            "Lower split_years or add longer maturities."
        )

    # Grid for plotting
    ttm_min = float(get_nested(cfg, ("grid", "ttm_min"), default=float(np.min(ttm_all))))
    ttm_max = float(get_nested(cfg, ("grid", "ttm_max"), default=float(np.max(ttm_all))))
    n_points = int(get_nested(cfg, ("grid", "n_points"), default=250))
    ttm_grid = np.linspace(ttm_min, ttm_max, n_points)

    # Build GP config from YAML (robust to param names)
    gp_cfg = get_nested(cfg, ("gp",), default={}) or {}

    kernel_type = gp_cfg.get("kernel_type", "rbf_periodic")
    n_restarts = int(gp_cfg.get("n_restarts_optimizer", 3))
    normalize_y = bool(gp_cfg.get("normalize_y", True))

    # Start from defaults
    model_cfg = GasForwardGPConfig(
        kernel_type=kernel_type,
        n_restarts_optimizer=n_restarts,
        normalize_y=normalize_y,
    )

    # rbf params
    rbf_cfg = gp_cfg.get("rbf", {}) or {}
    if "length_scale" in rbf_cfg:
        model_cfg.rbf_params = set_if_hasattr(model_cfg.rbf_params, "length_scale", float(rbf_cfg["length_scale"]))

    # periodic params
    per_cfg = gp_cfg.get("periodic", {}) or {}
    if "length_scale" in per_cfg:
        model_cfg.periodic_params = set_if_hasattr(model_cfg.periodic_params, "length_scale", float(per_cfg["length_scale"]))
    # period key variations
    period_val = parse_periodic_cfg(per_cfg)
    # try common attribute names
    for attr in ("period_years", "period", "periodicity"):
        model_cfg.periodic_params = set_if_hasattr(model_cfg.periodic_params, attr, float(period_val))

    # noise params (optional)
    # In your wrapper, WhiteKernel handles noise; keep default noise_params unless you exposed config keys.
    # (If you later add gp.noise.* keys, you can map them here.)

    # Fit model on TRAIN only
    model = GasForwardGP(config=model_cfg)
    print(f"[INFO] Fitting GP on train set (n={len(ttm_train)})...")
    model.fit(ttm_train, y_train)

    # Predict on grid + test points
    mean_grid, std_grid = model.predict(ttm_grid, return_std=True)
    mean_test, std_test = model.predict(ttm_test, return_std=True)

    # Metrics on TEST
    reg = regression_metrics(y_test, mean_test)
    prob = probabilistic_metrics(y_test, mean_test, std_test)

    print("\n=== Henry Hub — Temporal split metrics (TEST only) ===")
    print(f"Split @ {split_years:.2f}y | train n={len(ttm_train)} | test n={len(ttm_test)}")
    print(f"RMSE        : {reg.rmse:.4f}")
    print(f"MAE         : {reg.mae:.4f}")
    print(f"MAPE        : {reg.mape}")
    print(f"Coverage 95%: {prob.coverage_95:.3f}")
    print(f"Gaussian NLL: {prob.nll_gaussian:.4f}")
    print("=====================================================\n")

    # Output paths
    reports_dir_rel = get_nested(cfg, ("reports", "henry_hub_dir"), default="reports/henry_hub_results")
    reports_dir = PROJECT_ROOT / reports_dir_rel
    ensure_dir(reports_dir)

    out_png = reports_dir / "henry_hub_temporal_split.png"
    out_csv = reports_dir / "henry_hub_temporal_split_predictions.csv"

    # Save CSV (test preds)
    out_df = pd.DataFrame(
        {
            "ttm_years": ttm_test,
            "y_true": y_test,
            "y_pred_mean": mean_test,
            "y_pred_std": std_test,
        }
    ).sort_values("ttm_years")
    out_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved predictions: {out_csv}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.fill_between(
        ttm_grid,
        mean_grid - 1.96 * std_grid,
        mean_grid + 1.96 * std_grid,
        alpha=0.2,
        label="95% CI (fit on train)",
    )
    plt.plot(ttm_grid, mean_grid, linestyle="--", label="GP mean (fit on train)")

    plt.scatter(ttm_train, y_train, s=55, label="Train points")
    plt.scatter(ttm_test, y_test, s=70, label="Test points", marker="x")

    # show test predictions with uncertainty
    plt.errorbar(
        ttm_test,
        mean_test,
        yerr=1.96 * std_test,
        fmt="o",
        capsize=3,
        label="Test predictions (mean ± 1.96σ)",
    )

    plt.title("Henry Hub — Temporal split (train short maturities → test long maturities)")
    plt.xlabel("Time to maturity (years)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=300)
    print(f"[INFO] Saved figure: {out_png}")
    plt.show()

    print(f"[INFO] Fitted kernel:\n{model.kernel_}")


if __name__ == "__main__":
    main()
