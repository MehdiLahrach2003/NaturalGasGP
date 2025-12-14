from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    # naturalgasgp/cli.py -> project root is parent of "naturalgasgp"
    return Path(__file__).resolve().parents[1]


def _run_script(rel_path: str) -> int:
    """
    Run a Python script located in the repo using the current interpreter.
    This ensures the venv / editable install environment is used.
    """
    root = _project_root()
    script_path = root / rel_path

    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return 1

    cmd = [sys.executable, str(script_path)]
    print(f"[INFO] Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(root))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="naturalgasgp",
        description="Natural Gas Forward Curve modeling with Gaussian Processes",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("synthetic", help="Generate synthetic forward curve dataset")
    sub.add_parser("rbf", help="Fit GP with RBF kernel on synthetic dataset")
    sub.add_parser("seasonality", help="Fit GP with RBF + Periodic kernel on synthetic dataset")
    sub.add_parser("compare", help="Compare RBF vs seasonal GP on synthetic dataset")
    sub.add_parser("assets", help="Generate a full set of report assets (plots/metrics)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    mapping = {
        "synthetic": "experiments/generate_synthetic_curve.py",
        "rbf": "experiments/run_gp_rbf.py",
        "seasonality": "experiments/run_gp_seasonality.py",
        "compare": "experiments/compare_gp_models.py",
        "assets": "experiments/make_report_assets.py",
    }

    return _run_script(mapping[args.command])


if __name__ == "__main__":
    raise SystemExit(main())
