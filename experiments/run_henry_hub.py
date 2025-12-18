from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from gp_model.gas_forward_gp import GasForwardGP, GasForwardGPConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detect (ttm_years, y) columns in a robust way.
    We accept several naming conventions, then fallback to explicit error.
    """
    cols = {c.lower(): c for c in df.columns}

    ttm_candidates = ["ttm_years", "ttm", "maturity_years", "tenor_years", "time_to_maturity_years"]
    y_candidates = ["forward_observed", "forward_price", "price", "spot", "spot_price", "y"]

    ttm_col: Optional[str] = None
    y_col: Optional[str] = None

    for c in ttm_candidates:
        if c in cols:
            ttm_col = cols[c]
            break

    for c in y_candidates:
        if c in cols:
            y_col = cols[c]
            break

    if ttm_col is None or y_col is None:
        raise KeyError(
            "Impossible de détecter les colonnes. "
            f"Colonnes trouvées: {list(df.columns)}. "
            "Attendu: une colonne de temps type 'ttm_years' et une colonne prix type "
            "'forward_observed' ou 'forward_price' (ou équivalent)."
        )

    return ttm_col, y_col


def main() -> None:
    # --- Config ---
    cfg_path = PROJECT_ROOT / "config" / "default.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config introuvable: {cfg_path}")

    cfg = load_yaml(cfg_path)

    data_cfg = cfg.get("data", {})
    reports_cfg = cfg.get("reports", {})
    grid_cfg = cfg.get("grid", {})
    gp_cfg = cfg.get("gp", {})

    csv_rel = data_cfg.get("henry_hub_csv", "data/henry_hub/henry_hub.csv")
    csv_path = (PROJECT_ROOT / csv_rel).resolve()

    out_dir = PROJECT_ROOT / reports_cfg.get("henry_hub_dir", "reports/henry_hub_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    if not csv_path.exists():
        print(f"[INFO] Loading config: {cfg_path}")
        print(f"[WARN] Henry Hub CSV not found: {csv_path}")
        print("[HINT] Mets ton CSV ici (non commité): data/henry_hub/henry_hub.csv")
        print("[HINT] Il doit contenir au minimum: ttm_years + forward_price (ou forward_observed)")
        return

    print(f"[INFO] Loading config: {cfg_path}")
    print(f"[INFO] Loading Henry Hub CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    ttm_col, y_col = _detect_columns(df)
    print(f"[INFO] Using columns: ttm='{ttm_col}', y='{y_col}'")

    df = df[[ttm_col, y_col]].copy()
    df[ttm_col] = pd.to_numeric(df[ttm_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[ttm_col, y_col]).sort_values(ttm_col)

    if len(df) < 5:
        raise ValueError(f"Pas assez de points après nettoyage NaN (n={len(df)}).")

    ttm_obs = df[ttm_col].to_numpy(dtype=float)
    y_obs = df[y_col].to_numpy(dtype=float)

    # --- Grid for plotting ---
    ttm_min = float(grid_cfg.get("ttm_min", float(np.min(ttm_obs))))
    ttm_max = float(grid_cfg.get("ttm_max", float(np.max(ttm_obs))))
    n_points = int(grid_cfg.get("n_points", 250))
    ttm_grid = np.linspace(ttm_min, ttm_max, n_points)

    # --- Model ---
    # On utilise le kernel "complet" (RBF + periodic + noise) par défaut,
    # comme sur tes figures récentes.
    model_cfg = GasForwardGPConfig(
        kernel_type="rbf_periodic",
        n_restarts_optimizer=int(gp_cfg.get("n_restarts_optimizer", 3)),
        normalize_y=bool(gp_cfg.get("normalize_y", True)),
        random_state=int(gp_cfg.get("random_state", 42)),
    )
    model = GasForwardGP(model_cfg)

    print("[INFO] Fitting GP (RBF + periodic + noise)...")
    model.fit(ttm_obs, y_obs)

    mean, std = model.predict(ttm_grid, return_std=True)
    ci = 1.96 * std
    lo, hi = mean - ci, mean + ci

    # --- Plot ---
    out_path = out_dir / "henry_hub_gp_reconstruction.png"

    plt.figure(figsize=(12, 6))
    plt.plot(ttm_grid, mean, linestyle="--", label="GP mean (RBF + Periodic)")
    plt.fill_between(ttm_grid, lo, hi, alpha=0.2, label="95% confidence band")
    plt.scatter(ttm_obs, y_obs, s=40, label="Observed (Henry Hub)")

    plt.title("Henry Hub — GP Reconstruction (RBF + Periodic)")
    plt.xlabel("Time to maturity (years)")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Figure saved to: {out_path}")
    print(f"[INFO] Fitted kernel: {model.kernel_}")

    plt.show()


if __name__ == "__main__":
    main()
