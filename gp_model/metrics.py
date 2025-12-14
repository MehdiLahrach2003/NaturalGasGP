from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    """Simple regression metrics container."""
    rmse: float
    mae: float
    mape: float | None


@dataclass(frozen=True)
class ProbabilisticMetrics:
    """Uncertainty-aware metrics container."""
    coverage_95: float  # fraction of y_true within mean +/- 1.96*std
    nll_gaussian: float  # average Gaussian negative log-likelihood


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float | None:
    """
    Mean Absolute Percentage Error.
    Returns None if y_true contains too many near-zero values (unstable).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.maximum(np.abs(y_true), eps)
    # If too many values are effectively zero, MAPE becomes meaningless.
    if np.mean(denom <= eps) > 0.05:
        return None
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    return RegressionMetrics(
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        mape=mape(y_true, y_pred),
    )


def coverage_95(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    lo = mean - 1.96 * std
    hi = mean + 1.96 * std
    inside = (y_true >= lo) & (y_true <= hi)
    return float(np.mean(inside))


def gaussian_nll(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-9) -> float:
    """
    Average Gaussian negative log-likelihood.
    Assumes independent Normal(mean, std^2).
    """
    y_true = np.asarray(y_true, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    var = np.maximum(std ** 2, eps)
    return float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + (y_true - mean) ** 2 / var)))


def probabilistic_metrics(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> ProbabilisticMetrics:
    return ProbabilisticMetrics(
        coverage_95=coverage_95(y_true, mean, std),
        nll_gaussian=gaussian_nll(y_true, mean, std),
    )
