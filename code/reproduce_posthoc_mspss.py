from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True,
                        help="CSV with ID, MSPSS_total, and eight NET columns.")
    parser.add_argument("--network-definitions", type=Path, required=True)
    parser.add_argument("--permutation-scheme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fisher_ci(r_value: float, n: int) -> tuple[float, float]:
    z_value = np.arctanh(r_value)
    se = 1 / np.sqrt(n - 3)
    critical = stats.norm.ppf(0.975)
    return tuple(np.tanh([z_value - critical * se, z_value + critical * se]))


def main() -> int:
    args = parse_args()
    data = pd.read_csv(args.scores)
    definitions = pd.read_csv(args.network_definitions).set_index("network_id")
    orders = np.load(args.permutation_scheme)["global_orders"]
    if orders.shape != (10000, len(data)):
        raise ValueError(f"Unexpected permutation shape: {orders.shape}")
    y = data["MSPSS_total"].to_numpy(float)
    yz = (y - y.mean()) / y.std(ddof=0)
    columns = sorted(column for column in data if column.startswith("NET"))
    rows, null_columns = [], []
    for column in columns:
        x = data[column].to_numpy(float)
        xz = (x - x.mean()) / x.std(ddof=0)
        observed = float(np.mean(xz * yz))
        null = np.mean(yz[orders] * xz[None, :], axis=1)
        p_perm = (np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)
        low, high = fisher_ci(observed, len(x))
        definition = definitions.loc[column]
        rows.append({
            "network_id": column,
            "representation": definition["representation"],
            "local_outcome_origin": definition["local_outcome"],
            "analysis_status": "post_hoc_cross_outcome"
                if definition["local_outcome"] == "BAI"
                else "same_construct_external_main_effect",
            "n": len(x),
            "r_pearson": observed,
            "ci95_low_fisher": low,
            "ci95_high_fisher": high,
            "p_raw_two_sided": stats.pearsonr(x, y).pvalue,
            "p_permutation_two_sided": p_perm,
        })
        null_columns.append(null)
    result = pd.DataFrame(rows)
    result["q_BH_permutation_8"] = multipletests(
        result["p_permutation_two_sided"], method="fdr_bh"
    )[1]
    null_max = np.max(np.abs(np.column_stack(null_columns)), axis=1)
    result["p_maxT_8_vs_MSPSS"] = [
        (np.count_nonzero(null_max >= abs(value)) + 1) / (len(null_max) + 1)
        for value in result["r_pearson"]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
