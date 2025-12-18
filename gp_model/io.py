from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


# ---------------------------
# YAML
# ---------------------------

def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data)}")
    return data


# ---------------------------
# Curve CSV loading
# ---------------------------

@dataclass(frozen=True)
class CurveData:
    ttm_years: np.ndarray
    y: np.ndarray
    df: pd.DataFrame  # cleaned df (useful for debug)


def _coerce_float_series(s: pd.Series) -> pd.Series:
    # Handles commas, strings, etc.
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def load_curve_csv(
    csv_path: str | Path,
    ttm_col: str,
    y_col: str,
    dropna: bool = True,
) -> CurveData:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if ttm_col not in df.columns:
        raise KeyError(f"Missing column '{ttm_col}' in {csv_path}. Columns: {list(df.columns)}")
    if y_col not in df.columns:
        raise KeyError(f"Missing column '{y_col}' in {csv_path}. Columns: {list(df.columns)}")

    df = df.copy()
    df[ttm_col] = _coerce_float_series(df[ttm_col])
    df[y_col] = _coerce_float_series(df[y_col])

    if dropna:
        df = df.dropna(subset=[ttm_col, y_col]).reset_index(drop=True)

    ttm = df[ttm_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    if len(ttm) == 0:
        raise ValueError(f"No usable rows after cleaning (all NaN?) for {csv_path}")

    return CurveData(ttm_years=ttm, y=y, df=df)


def load_synthetic_curve(
    csv_path: str | Path,
    ttm_col: str = "ttm_years",
    y_col: str = "forward_observed",
) -> CurveData:
    return load_curve_csv(csv_path, ttm_col=ttm_col, y_col=y_col, dropna=True)


def load_henry_hub_curve(
    csv_path: str | Path,
    ttm_col: str,
    y_col: str,
) -> CurveData:
    # For real data we *always* drop NaNs (scikit-learn GP refuses NaNs)
    return load_curve_csv(csv_path, ttm_col=ttm_col, y_col=y_col, dropna=True)
