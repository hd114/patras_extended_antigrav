#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path


def parse_fhe_metrics_from_file(txt_path: Path) -> dict:
    """
    Parse macro precision, macro recall, weighted F1 and macro ROC-AUC
    from a given FHE evaluation text file.

    Expected lines in the file:
        Prec(m): 0.5784
        Rec(m): 0.6313
        F1(w): 0.6222
        ROC-AUC(m): 0.9814
    """
    metrics = {
        "prec_m": None,
        "rec_m": None,
        "f1w_fhe": None,
        "roc_auc_m": None,
    }

    number_pattern = r"([0-9]+(?:\.[0-9]+)?)"

    patterns = {
        "prec_m": re.compile(rf"^Prec\(m\):\s*{number_pattern}"),
        "rec_m": re.compile(rf"^Rec\(m\):\s*{number_pattern}"),
        "f1w_fhe": re.compile(rf"^F1\(w\):\s*{number_pattern}"),
        "roc_auc_m": re.compile(rf"^ROC-AUC\(m\):\s*{number_pattern}"),
    }

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            for key, pattern in patterns.items():
                match = pattern.match(line)
                if match:
                    metrics[key] = float(match.group(1))

    missing = [k for k, v in metrics.items() if v is None]
    if missing:
        raise ValueError(
            f"Could not find all FHE metrics in file {txt_path}. Missing: {', '.join(missing)}"
        )

    return metrics


def parse_torch_f1_from_file(txt_path: Path) -> float:
    """
    Parse weighted F1 from a torch QAT evaluation text file.

    Expected line in the file:
        F1(w): 0.6452
    """
    number_pattern = r"([0-9]+(?:\.[0-9]+)?)"
    pattern = re.compile(rf"^F1\(w\):\s*{number_pattern}")

    f1w_torch = None

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = pattern.match(line)
            if match:
                f1w_torch = float(match.group(1))
                break

    if f1w_torch is None:
        raise ValueError(
            f"Could not find weighted F1 in torch eval file {txt_path}"
        )

    return f1w_torch


def extract_experiment_info(fhe_txt_path: Path) -> dict:
    """
    Extract dataset, quant_bits and unpruned_neurons from the path and file name.

    Assumptions:
        - Dataset is the token after 'QAT_' in the file name, for example:
          eval_fhe_QAT_CICIoT_100h_qb3_ep53_...
            -> dataset = 'CICIoT'
        - quant_bits is the number after 'qb', searched in the full path, for example:
          'qb3' -> quant_bits = 3
        - unpruned_neurons is the number after 'p', searched in the full path,
          for example directory 'qb3_p16_unstructured_C' -> unpruned_neurons = 16
    """
    name = fhe_txt_path.name

    dataset_match = re.search(r"QAT_([A-Za-z0-9]+)_", name)
    if not dataset_match:
        raise ValueError(f"Could not extract dataset from file name {name}")
    dataset = dataset_match.group(1)

    path_str = str(fhe_txt_path)

    qb_match = re.search(r"qb(\d+)", path_str)
    if not qb_match:
        raise ValueError(f"Could not extract quant_bits from path {path_str}")
    quant_bits = int(qb_match.group(1))

    p_match = re.search(r"[_-]p(\d+)", path_str)
    if not p_match:
        raise ValueError(f"Could not extract unpruned_neurons from path {path_str}")
    unpruned_neurons = int(p_match.group(1))

    if dataset.lower().startswith("edge"):
        dataset_norm = "EdgeIIoT"
    elif dataset.lower().startswith("ege"):
        dataset_norm = "EdgeIIoT"
    else:
        dataset_norm = dataset

    return {
        "dataset": dataset_norm,
        "quant_bits": quant_bits,
        "unpruned_neurons": unpruned_neurons,
    }


def find_torch_eval_file(fhe_txt_path: Path) -> Path:
    """
    Find the corresponding eval_torch_QAT_*_pytorch_eval.txt file
    in the same directory as the given FHE txt file.
    """
    directory = fhe_txt_path.parent
    candidates = list(directory.glob("eval_torch_QAT_*_pytorch_eval.txt"))

    if not candidates:
        raise FileNotFoundError(
            f"No eval_torch_QAT_*_pytorch_eval.txt found in {directory}"
        )

    if len(candidates) == 1:
        return candidates[0]

    fhe_info = extract_experiment_info(fhe_txt_path)
    dataset = fhe_info["dataset"]

    for cand in candidates:
        if f"QAT_{dataset}_" in cand.name:
            return cand

    return candidates[0]


def gather_all_results(root_dir: Path, output_csv_path: Path) -> Path:
    """
    Walk through all qb*_p*_* folders under root_dir (for example
    qb3_p16_unstructured_C or qb7_p16_structured_E) and collect
    metrics for every rep_* subfolder.

    For each rep_* folder, this function expects:
        - one FHE eval file: eval_fhe_QAT_*_fhe_eval_simulate_simulate.txt
        - one corresponding torch eval file: eval_torch_QAT_*_pytorch_eval.txt

    It writes a single aggregated CSV with one row per rep_* folder.

    Columns (in this order):
        config_folder
        dataset
        quant_bits
        unpruned_neurons
        f1w_torch
        f1w_fhe
        prec_m_fhe
        rec_m_fhe
        roc_auc_m_fhe
        repetition
    """
    rows = []

    # Changed pattern here: was "qb*_p*_unstructured_*"
    for config_dir in sorted(root_dir.glob("qb*_p*_*")):
        if not config_dir.is_dir():
            continue

        config_name = config_dir.name
        print(f"Processing config folder: {config_name}")

        rep_dirs = sorted(config_dir.glob("rep_*"))
        if not rep_dirs:
            print(f"Warning: no rep_* subfolders found in {config_dir}")
            continue

        for rep_dir in rep_dirs:
            if not rep_dir.is_dir():
                continue

            rep_name = rep_dir.name
            rep_match = re.match(r"rep_(\d+)", rep_name)
            if not rep_match:
                print(f"Warning: could not parse repetition from folder {rep_name}")
                continue
            repetition = int(rep_match.group(1))

            fhe_candidates = list(
                rep_dir.glob("eval_fhe_QAT_*_fhe_eval_simulate_simulate.txt")
            )
            if not fhe_candidates:
                print(
                    f"Warning: no eval_fhe_QAT_*_fhe_eval_simulate_simulate.txt "
                    f"found in {rep_dir}"
                )
                continue
            if len(fhe_candidates) > 1:
                print(
                    f"Warning: multiple FHE eval files found in {rep_dir}, "
                    f"using first one: {fhe_candidates[0].name}"
                )

            fhe_txt_path = fhe_candidates[0]

            try:
                fhe_metrics = parse_fhe_metrics_from_file(fhe_txt_path)
                exp_info = extract_experiment_info(fhe_txt_path)
                torch_txt_path = find_torch_eval_file(fhe_txt_path)
                torch_f1w = parse_torch_f1_from_file(torch_txt_path)
            except Exception as exc:
                print(f"Error processing {rep_dir}: {exc}")
                continue

            row = {
                "config_folder": config_name,
                "dataset": exp_info["dataset"],
                "quant_bits": exp_info["quant_bits"],
                "unpruned_neurons": exp_info["unpruned_neurons"],
                "f1w_torch": torch_f1w,
                "f1w_fhe": fhe_metrics["f1w_fhe"],
                "prec_m_fhe": fhe_metrics["prec_m"],
                "rec_m_fhe": fhe_metrics["rec_m"],
                "roc_auc_m_fhe": fhe_metrics["roc_auc_m"],
                "repetition": repetition,
            }

            rows.append(row)

    if not rows:
        raise RuntimeError(f"No results collected under {root_dir}")

    # Sort rows ascending:
    # first by dataset, then by quant_bits, then by unpruned_neurons
    rows.sort(
        key=lambda r: (r["dataset"], r["quant_bits"], r["unpruned_neurons"])
    )

    fieldnames = [
        "config_folder",
        "dataset",
        "quant_bits",
        "unpruned_neurons",
        "f1w_torch",
        "f1w_fhe",
        "prec_m_fhe",
        "rec_m_fhe",
        "roc_auc_m_fhe",
        "repetition",
    ]

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Aggregated CSV written to: {output_csv_path}")
    print(f"Total rows: {len(rows)}")

    return output_csv_path


def main() -> None:
    """
    Main entry point.

    Usage examples (from FHE_athens):

        1) Use default meta run directory and output path:
            python extract_qat_metrics_to_csv.py

        2) Specify a different meta run directory:
            python extract_qat_metrics_to_csv.py results/qat_runs/meta_run_.../

        3) Specify meta run directory and explicit output CSV:
            python extract_qat_metrics_to_csv.py results/qat_runs/meta_run_.../ my_results.csv
    """
    if len(sys.argv) == 1:
        root_dir = Path(
            "results/qat_runs/meta_run_20251129_191433_q357_p16_new_unstr"
        )
        output_csv_path = root_dir / "AGGREGATED_RESULTS.csv"
    elif len(sys.argv) == 2:
        root_dir = Path(sys.argv[1])
        output_csv_path = root_dir / "AGGREGATED_RESULTS.csv"
    else:
        root_dir = Path(sys.argv[1])
        output_csv_path = Path(sys.argv[2])

    if not root_dir.is_dir():
        print(f"Error: root directory does not exist or is not a directory: {root_dir}")
        sys.exit(1)

    print(f"Root directory: {root_dir}")
    print(f"Output CSV: {output_csv_path}")

    try:
        gather_all_results(root_dir=root_dir, output_csv_path=output_csv_path)
    except Exception as exc:
        print(f"Failed to aggregate results: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
