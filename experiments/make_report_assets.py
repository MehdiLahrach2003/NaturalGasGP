from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_script(rel_path: str) -> None:
    script_path = (PROJECT_ROOT / rel_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    print(f"\n[ASSETS] Running: {script_path}")
    subprocess.check_call([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))


def main() -> None:
    # ---- Synthetic pipeline ----
    run_script("experiments/run_gp_rbf.py")
    run_script("experiments/run_gp_seasonality.py")
    run_script("experiments/run_gp_full_kernel.py")
    run_script("experiments/compare_gp_models.py")

    # ---- Henry Hub pipeline ----
    run_script("experiments/run_henry_hub.py")
    run_script("experiments/henry_hub_metrics.py")
    run_script("experiments/henry_hub_kernel_ablation.py")

    print("\n[ASSETS] Done. All figures/CSVs refreshed.")


if __name__ == "__main__":
    main()
