import sys
from pathlib import Path

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from gp_model.gas_forward_gp import GasForwardGP, GasForwardGPConfig
from gp_model.metrics import regression_metrics, probabilistic_metrics
from gp_model.plotting import CurvePlotSpec, plot_gp_reconstruction, plot_bar_comparison
from gp_model.io import make_run_id, write_metadata, write_json


def main():
    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    data_path = PROJECT_ROOT / "data" / "synthetic" / "synthetic_forward_curve.csv"
    base_reports_dir = PROJECT_ROOT / "reports" / "synthetic_results"
    base_reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = make_run_id(prefix="synthetic_assets")
    run_dir = base_reports_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    write_metadata(run_dir / "run_metadata.json", run_id=run_id)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    df = pd.read_csv(data_path)

    ttm = df["ttm_years"].to_numpy(dtype=float)
    forward_true = df["forward_true"].to_numpy(dtype=float)

    mask_obs = df["is_observed"].to_numpy(dtype=bool)
    ttm_obs = ttm[mask_obs]
    forward_obs = df["forward_observed"].to_numpy(dtype=float)[mask_obs]

    # (scikit-learn GP expects 2D X)
    X = ttm.reshape(-1, 1)
    X_obs = ttm_obs.reshape(-1, 1)

    # ------------------------------------------------------------
    # Define models
    # ------------------------------------------------------------
    models = {
        "RBF": GasForwardGP(GasForwardGPConfig(kernel_type="rbf")),
        "RBF+Periodic": GasForwardGP(GasForwardGPConfig(kernel_type="rbf_periodic")),
    }

    rows = []
    preds = {}

    for name, model in models.items():
        print(f"\n=== Fitting {name} ===")
        model.fit(X_obs, forward_obs)
        mean_pred, std_pred = model.predict(X, return_std=True)

        preds[name] = (mean_pred, std_pred)

        reg = regression_metrics(forward_true, mean_pred)
        prob = probabilistic_metrics(forward_true, mean_pred, std_pred)

        rows.append({
            "model": name,
            "rmse": reg.rmse,
            "mae": reg.mae,
            "mape": reg.mape if reg.mape is not None else np.nan,
            "coverage_95": prob.coverage_95,
            "nll_gaussian": prob.nll_gaussian,
            "avg_std": float(np.mean(std_pred)),
        })

        print(f"RMSE={reg.rmse:.4f} | MAE={reg.mae:.4f} | Cov95={prob.coverage_95:.3f} | NLL={prob.nll_gaussian:.4f}")

    metrics_df = pd.DataFrame(rows).sort_values("rmse")
    metrics_path = run_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved: {metrics_path}")

    # Also save a simple JSON summary
    write_json(run_dir / "metrics.json", {"rows": rows})

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    # Reconstruction plots per model
    for name, (mean_pred, std_pred) in preds.items():
        out_path = run_dir / f"reconstruction_{name.lower()}.png"
        plot_gp_reconstruction(
            ttm=ttm,
            forward_true=forward_true,
            ttm_obs=ttm_obs,
            forward_obs=forward_obs,
            mean_pred=mean_pred,
            std_pred=std_pred,
            spec=CurvePlotSpec(title=f"GP reconstruction (synthetic) — {name}"),
            out_path=out_path,
        )
        print(f"Saved: {out_path}")

    # Bar comparisons
    labels = list(models.keys())
    rmse_vals = [float(metrics_df.set_index("model").loc[l, "rmse"]) for l in labels]
    avgstd_vals = [float(metrics_df.set_index("model").loc[l, "avg_std"]) for l in labels]

    plot_bar_comparison(
        labels=labels,
        values=rmse_vals,
        title="RMSE comparison (synthetic)",
        ylabel="RMSE",
        out_path=run_dir / "rmse_comparison.png",
    )
    plot_bar_comparison(
        labels=labels,
        values=avgstd_vals,
        title="Average predictive uncertainty (synthetic)",
        ylabel="Avg std",
        out_path=run_dir / "uncertainty_comparison.png",
    )

    print(f"\nAll assets saved under:\n{run_dir}\n")


if __name__ == "__main__":
    main()
