from __future__ import annotations

import numpy as np
import pandas as pd

from gp_model.gas_forward_gp import GasForwardGP, GasForwardGPConfig


def test_gp_fit_predict_smoke() -> None:
    # Simple synthetic toy dataset (not the full pipeline)
    ttm = np.array([0.25, 0.50, 1.0, 1.5, 2.0, 3.0], dtype=float)
    y = np.array([2.40, 2.20, 2.80, 2.55, 3.10, 2.90], dtype=float)

    # Fit GP
    cfg = GasForwardGPConfig()
    model = GasForwardGP(config=cfg)
    model.fit(ttm, y)

    # Predict on a grid
    grid = np.linspace(0.25, 3.0, 25)
    mean, std = model.predict(grid, return_std=True)

    # Basic sanity checks
    assert mean.shape == grid.shape
    assert std.shape == grid.shape
    assert np.all(std >= 0.0)


def test_metrics_available() -> None:
    # Optional: ensure metrics module exists and can be imported
    import gp_model.metrics  # noqa: F401
