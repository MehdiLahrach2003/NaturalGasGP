# Project notes — NaturalGasGP

## What this repo does
Reconstructs a natural gas forward curve as a function of time-to-maturity (TTM, in years) using Gaussian Process regression.
Two main settings:
- Synthetic curve (for controlled evaluation)
- Henry Hub example (real-ish demo with sparse points)

## Data conventions
Expected minimum columns:
- `ttm_years`: time to maturity in years
- `forward_observed` OR `forward_price`: observed forward price

Synthetic data lives in:
- `data/synthetic/synthetic_forward_curve.csv`

Henry Hub data (not committed) should be placed at:
- `data/henry_hub/henry_hub.csv`

## Main commands
After install (editable recommended):
- `pip install -e .`

Run pipeline:
- `naturalgasgp assets`

This regenerates figures/CSVs under `reports/`.

## Modeling choices
- Baseline kernel: RBF + WhiteKernel
- Seasonal kernel: RBF + ExpSineSquared + WhiteKernel
- Diagnostics:
  - In-sample metrics (RMSE/MAE/MAPE + coverage95 + Gaussian NLL)
  - LOOCV diagnostics
  - Kernel ablation study

## Known limitations
- The "true curve" is available only in synthetic experiments.
- Henry Hub experiments are evaluated mainly via LOOCV and probabilistic metrics.
- Data ingestion expects clean numeric columns; NaNs are dropped.
