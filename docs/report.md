# Gaussian Process Modeling of Natural Gas Forward Curves

## Abstract
This project reconstructs natural gas forward curves as a function of time-to-maturity (TTM) using Gaussian Process (GP) regression.
We evaluate a smooth baseline model (RBF + noise) and a seasonal model (RBF + periodic + noise), and we report both point-wise
and uncertainty-aware diagnostics. The pipeline is fully reproducible via a CLI command that regenerates all figures and CSV outputs.

## Introduction
Forward curves are typically observed at a sparse set of maturities and can be noisy. A GP provides a principled non-parametric model
to interpolate the curve while quantifying uncertainty. We focus on:
1) curve reconstruction, 2) seasonality modeling, and 3) model comparison using validation diagnostics.

## Data

### Synthetic dataset
A controlled synthetic forward curve is generated to validate reconstruction quality.
We produce:
- a smooth underlying curve (treated as "true"),
- sparse observations at selected maturities,
- optional observation noise.

File:
- `data/synthetic/synthetic_forward_curve.csv`

Expected columns:
- `ttm_years`
- `true_curve` (optional, synthetic-only)
- `forward_observed` (noisy sparse observations)

### Henry Hub dataset
A small Henry Hub example is used as a real-data-style sanity check for the modeling pipeline.
The file is not committed by default.

File path expected:
- `data/henry_hub/henry_hub.csv`

Expected columns:
- `ttm_years`
- `forward_price` (or `forward_observed`)

## Methodology

### Gaussian Process regression
We model the forward price function \( f(t) \) where \( t \) is the time-to-maturity (years).
Given observations \( (t_i, y_i) \), a GP prior \( f \sim \mathcal{GP}(0, k(\cdot,\cdot)) \) yields a posterior predictive distribution
\( f(t_\*) \mid \mathcal{D} \sim \mathcal{N}(\mu(t_\*), \sigma^2(t_\*)) \).

Uncertainty is reported via:
- 95% band: \( \mu \pm 1.96 \sigma \)

### Kernels
We compare:
- **RBF + WhiteKernel**  
  Smoothness-only baseline.
- **RBF + ExpSineSquared + WhiteKernel**  
  Captures smooth trend + seasonality.

### Metrics
We compute:
- RMSE, MAE, (MAPE when stable)
- Coverage@95%: fraction of points within \( \mu \pm 1.96\sigma \)
- Gaussian NLL: average negative log-likelihood under \( \mathcal{N}(\mu, \sigma^2) \)

For Henry Hub, we also run **LOOCV** (leave-one-out cross-validation) and report LOOCV diagnostics:
- LOOCV RMSE
- average predictive std under LOOCV

## Experiments

### Step 1 — Data preparation / standardization
- Ensure consistent CSV columns and robust loading.
- Generate synthetic data and define the evaluation grid.

### Step 2 — Baseline GP reconstruction (RBF)
Fit the baseline model and plot:
- observations vs reconstructed curve
- uncertainty band

### Step 3 — Seasonality GP (RBF + periodic)
Fit the seasonal model and compare to baseline.

### Henry Hub pipeline
We replicate the pipeline on Henry Hub-style data:
- reconstruction plot
- in-sample metrics (for sanity)
- LOOCV diagnostics (main evaluation)
- kernel ablation study

## Results
Qualitatively:
- Seasonal kernels often improve reconstruction when the curve exhibits periodic structure.
- Uncertainty grows in regions with fewer observations, as expected.
- LOOCV gives a more realistic measure than in-sample metrics for sparse real-data-like curves.

Artifacts produced by the pipeline are saved under:
- `reports/synthetic_results/`
- `reports/henry_hub_results/`

## Discussion
The seasonal kernel may reduce point error but can also change predictive uncertainty depending on how well periodicity is supported by the data.
LOOCV is critical to avoid overly optimistic in-sample conclusions, especially for small datasets.

## Conclusion
This repository provides a clean, reproducible GP pipeline to reconstruct forward curves with uncertainty quantification, with both synthetic validation
and a Henry Hub example. The CLI entry point regenerates all figures and metrics outputs.

## How to run
Editable install:
```bash
pip install -e .
