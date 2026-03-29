import numpy as np
import pandas as pd


def spc(model_xings, graphviz_xings):
    """
    Symmetric Percent Change of model vs Graphviz baseline.

    SPC = 100% * (1/Nt) * sum( (D_i - G_i) / max(D_i, G_i) )

    Negative SPC means model is better than Graphviz.
    Ranges from -100% to +100%.

    Args:
        model_xings:    array-like of crossing counts from your model (D_i)
        graphviz_xings: array-like of crossing counts from Graphviz baseline (G_i)

    Returns:
        SPC as a float percentage
    """
    D = np.array(model_xings, dtype=float)
    G = np.array(graphviz_xings, dtype=float)
    denom = np.maximum(D, G)
    # Both zero => no crossings either way, contributes 0
    safe_denom = np.where(denom > 0, denom, 1.0)
    terms = np.where(denom > 0, (D - G) / safe_denom, 0.0)
    return 100.0 * terms.mean()


def report_spc(csv_path="baseline_metrics.csv", baseline="neato_xing"):
    """
    Compute SPC for all models in the CSV relative to a chosen Graphviz baseline.

    Args:
        csv_path: path to metrics CSV (must have graph_id + crossing columns)
        baseline: column name to use as Graphviz reference (default: neato_xing)
    """
    df = pd.read_csv(csv_path).dropna()

    model_cols = [c for c in df.columns if c not in ("graph_id", "status") and c != baseline]

    print(f"SPC relative to '{baseline}' on {len(df)} graphs:\n")
    results = {}
    for col in model_cols:
        val = spc(df[col], df[baseline])
        results[col] = val
        marker = " <-- better" if val < 0 else ""
        print(f"  {col:25s}: {val:+.2f}%{marker}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute SPC metrics from a crossings CSV")
    parser.add_argument("--csv", type=str, default="baseline_metrics.csv")
    parser.add_argument("--baseline", type=str, default="neato_xing",
                        help="Column to use as Graphviz reference")
    args = parser.parse_args()

    report_spc(args.csv, args.baseline)
