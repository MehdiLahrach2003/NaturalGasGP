"""
Command-line interface (CLI) to run the project end-to-end.

Usage examples (from project root, with venv activated):
    python -m naturalgasgp.cli synthetic
    python -m naturalgasgp.cli rbf
    python -m naturalgasgp.cli seasonality
    python -m naturalgasgp.cli compare
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path


def _project_root() -> Path:
    # This file is: <root>/naturalgasgp/cli.py
    return Path(__file__).resolve().parents[1]


def _run_script(rel_path: str) -> None:
    """
    Execute a Python script as if it were run directly.
    This avoids import-path issues and keeps experiments runnable.
    """
    script_path = _project_root() / rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    runpy.run_path(str(script_path), run_name="__main__")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="naturalgasgp",
        description="Natural Gas Forward Curve modeling with Gaussian Processes",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("synthetic", help="Generate synthetic forward curve dataset")
    sub.add_parser("rbf", help="Fit GP with RBF + noise kernel on synthetic data")
    sub.add_parser("seasonality", help="Compare RBF vs RBF+Periodic on synthetic data")
    sub.add_parser("compare", help="Compare GP models and produce summary figures")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "synthetic":
        _run_script("experiments/generate_synthetic_curve.py")
    elif args.command == "rbf":
        _run_script("experiments/run_gp_rbf.py")
    elif args.command == "seasonality":
        _run_script("experiments/run_gp_seasonality.py")
    elif args.command == "compare":
        _run_script("experiments/compare_gp_models.py")
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
