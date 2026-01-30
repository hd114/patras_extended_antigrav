#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd


def compute_fhe_stats(input_csv: Path, output_csv: Path) -> None:
    """
    Read an aggregated results CSV, compute mean and std for all FHE metrics
    per config, and write them to a new CSV.

    Expected columns in input CSV:
        config_folder
        dataset
        quant_bits
        unpruned_neurons
        f1w_fhe
        prec_m_fhe
        rec_m_fhe
        roc_auc_m_fhe
    """
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = [
        "config_folder",
        "dataset",
        "quant_bits",
        "unpruned_neurons",
        "f1w_fhe",
        "prec_m_fhe",
        "rec_m_fhe",
        "roc_auc_m_fhe",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {', '.join(missing)}"
        )

    group_cols = ["config_folder", "dataset", "quant_bits", "unpruned_neurons"]
    fhe_cols = ["f1w_fhe", "prec_m_fhe", "rec_m_fhe", "roc_auc_m_fhe"]

    grouped = df.groupby(group_cols)[fhe_cols].agg(["mean", "std"]).reset_index()

    # Flatten MultiIndex columns, e.g. f1w_fhe_mean, f1w_fhe_std
    flat_cols = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            base, stat = col
            if stat == "" or base in group_cols:
                flat_cols.append(base)
            else:
                flat_cols.append(f"{base}_{stat}")
        else:
            flat_cols.append(col)

    grouped.columns = flat_cols

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False)

    print(f"Read from: {input_csv}")
    print(f"Wrote FHE stats to: {output_csv}")


def main() -> None:
    """
    Usage from FHE_athens:

        1) Default: use AGGREGATED_RESULTS.csv in current directory
           python aggregate_fhe_stats.py

        2) From FHE_athens, pass the meta_run folder:
           python aggregate_fhe_stats.py results/qat_runs/meta_run_20251129_093245_q357_p16_new

           This will read:
               results/qat_runs/meta_run_20251129_093245_q357_p16_new/AGGREGATED_RESULTS.csv
           and write:
               results/qat_runs/meta_run_20251129_093245_q357_p16_new/AGGREGATED_RESULTS_FHE_STATS.csv

        3) You can still pass a CSV directly:
           python aggregate_fhe_stats.py path/to/AGGREGATED_RESULTS.csv

        4) Or CSV + explicit output:
           python aggregate_fhe_stats.py path/to/AGGREGATED_RESULTS.csv path/to/output.csv
    """
    if len(sys.argv) == 1:
        # No arguments: assume AGGREGATED_RESULTS.csv in current directory
        input_csv = Path("AGGREGATED_RESULTS.csv")
        output_csv = input_csv.with_name("AGGREGATED_RESULTS_FHE_STATS.csv")
    else:
        arg1 = Path(sys.argv[1])

        if arg1.is_dir():
            # Argument is a meta_run directory
            meta_dir = arg1
            input_csv = meta_dir / "AGGREGATED_RESULTS.csv"
            if len(sys.argv) >= 3:
                output_csv = Path(sys.argv[2])
            else:
                output_csv = meta_dir / "AGGREGATED_RESULTS_FHE_STATS.csv"
        else:
            # Argument is a CSV file path
            input_csv = arg1
            if len(sys.argv) >= 3:
                output_csv = Path(sys.argv[2])
            else:
                output_csv = input_csv.with_name("AGGREGATED_RESULTS_FHE_STATS.csv")

    compute_fhe_stats(input_csv=input_csv, output_csv=output_csv)


if __name__ == "__main__":
    main()
