from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
import numpy as np


@dataclass(frozen=True)
class CleanXY:
    """Container for cleaned 1D x/y arrays."""
    x: np.ndarray
    y: np.ndarray


def as_1d_float(x: Iterable[float] | np.ndarray) -> np.ndarray:
    """Convert input to a 1D float numpy array."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


def drop_nan_pairs(x: np.ndarray, y: np.ndarray) -> CleanXY:
    """
    Drop rows where x or y is NaN/Inf.
    Returns cleaned x,y with same length.
    """
    x = as_1d_float(x)
    y = as_1d_float(y)
    mask = np.isfinite(x) & np.isfinite(y)
    return CleanXY(x=x[mask], y=y[mask])


def sort_by_x(x: np.ndarray, y: np.ndarray) -> CleanXY:
    """Sort pairs (x,y) by x ascending."""
    x = as_1d_float(x)
    y = as_1d_float(y)
    idx = np.argsort(x)
    return CleanXY(x=x[idx], y=y[idx])


def ensure_min_points(x: np.ndarray, y: np.ndarray, n_min: int = 3) -> None:
    """Raise if not enough points."""
    if len(x) < n_min:
        raise ValueError(f"Need at least {n_min} points, got {len(x)}")


def clean_sort_xy(x: Iterable[float] | np.ndarray, y: Iterable[float] | np.ndarray, n_min: int = 3) -> CleanXY:
    """
    Common utility:
    - cast to float
    - drop NaN/Inf pairs
    - sort by x
    - check min points
    """
    c = drop_nan_pairs(np.asarray(x), np.asarray(y))
    c = sort_by_x(c.x, c.y)
    ensure_min_points(c.x, c.y, n_min=n_min)
    return c


def make_grid(xmin: float, xmax: float, n_points: int) -> np.ndarray:
    """Create a 1D grid in [xmin, xmax]."""
    if n_points <= 1:
        raise ValueError("n_points must be >= 2")
    if xmax <= xmin:
        raise ValueError("xmax must be > xmin")
    return np.linspace(float(xmin), float(xmax), int(n_points))


def mean_std_to_ci95(mean: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) arrays for a 95% CI under Normal(mean, std^2)."""
    mean = as_1d_float(mean)
    std = as_1d_float(std)
    lo = mean - 1.96 * std
    hi = mean + 1.96 * std
    return lo, hi
