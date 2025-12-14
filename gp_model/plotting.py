from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class CurvePlotSpec:
    """Standard plot options for forward curve reconstruction."""
    title: str
    xlabel: str = "Time to maturity (years)"
    ylabel: str = "Forward price (USD/MMBtu)"
    show_grid: bool = True


def plot_gp_reconstruction(
    ttm: np.ndarray,
    forward_true: np.ndarray | None,
    ttm_obs: np.ndarray,
    forward_obs: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    spec: CurvePlotSpec,
    out_path: Path | None = None,
) -> None:
    """
    Plot forward curve reconstruction with 95% confidence band.
    - ttm: grid (1D)
    - forward_true: optional ground truth (1D)
    - ttm_obs, forward_obs: observed sparse points
    - mean_pred, std_pred: GP predictions on ttm grid
    """
    ttm = np.asarray(ttm, dtype=float).ravel()
    ttm_obs = np.asarray(ttm_obs, dtype=float).ravel()
    forward_obs = np.asarray(forward_obs, dtype=float).ravel()
    mean_pred = np.asarray(mean_pred, dtype=float).ravel()
    std_pred = np.asarray(std_pred, dtype=float).ravel()

    plt.figure(figsize=(11, 5))

    if forward_true is not None:
        forward_true = np.asarray(forward_true, dtype=float).ravel()
        plt.plot(ttm, forward_true, lw=2, label="True curve")

    plt.scatter(ttm_obs, forward_obs, s=45, label="Observed", zorder=3)

    plt.plot(ttm, mean_pred, lw=2, linestyle="--", label="GP mean")
    plt.fill_between(
        ttm,
        mean_pred - 1.96 * std_pred,
        mean_pred + 1.96 * std_pred,
        alpha=0.25,
        label="95% CI",
    )

    plt.title(spec.title)
    plt.xlabel(spec.xlabel)
    plt.ylabel(spec.ylabel)
    if spec.show_grid:
        plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)

    plt.show()


def plot_bar_comparison(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    out_path: Path | None = None,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y")
    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)

    plt.show()
