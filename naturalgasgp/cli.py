from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_assets() -> None:
    script = PROJECT_ROOT / "experiments" / "make_report_assets.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing assets script: {script}")

    cmd = [sys.executable, str(script)]
    print(f"[CLI] Running assets pipeline:\n  {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="naturalgasgp",
        description="NaturalGasGP – Gaussian Process pipeline for gas forwards",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- assets ----
    sub.add_parser(
        "assets",
        help="Run full pipeline and regenerate all figures/CSVs",
    )

    args = parser.parse_args()

    if args.command == "assets":
        run_assets()
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
