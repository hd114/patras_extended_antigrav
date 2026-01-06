import os
import sys
import pandas as pd
import subprocess
import time
import glob
import re
import argparse
from multiprocessing import Pool

# --- BASIS KONFIGURATION -----------------------------------------------------
META_RUN_FOLDER = "results/qat_runs/meta_run_20251129_234515_q357_p4_new_unstr"
CORES_PER_JOB = 4       # Strikt 4 Kerne pro Job ("Double-Lock")
MAX_PARALLEL_JOBS = 40  # Obergrenze

# --- FILTER (Manuell im Code) ---
# Hier kannst du exakte Ordnernamen eintragen. Hat VORRANG vor --filter.
# Beispiel: TARGET_CONFIGS = ["qb3_p8_structured_C", "qb3_p4_structured_C"]
TARGET_CONFIGS = ["qb3_p4_unstructured_C", "qb5_p4_unstructured_C"] 


# Pfade (Hardcoded)
EDGE_DATA_PATH = "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
CIC_DATA_PATH = "/home/jovyan/TenSEAL_projects/ciciot_dataset_all.npz"
SAMPLES = 10
# -----------------------------------------------------------------------------


def find_summary_csv(meta_folder):
    """Findet die neueste Summary CSV."""
    csvs = glob.glob(os.path.join(meta_folder, "meta_experiment_summary_*.csv"))
    if not csvs:
        fallback = os.path.join(meta_folder, "meta_experiment_summary.csv")
        if os.path.exists(fallback):
            return fallback
        return None
    return max(csvs, key=os.path.getmtime)


def get_tasks_from_summary(meta_folder):
    """Erstellt Task-Liste aus der Summary CSV."""
    summary_csv = find_summary_csv(meta_folder)
    if not summary_csv:
        print("[FEHLER] Keine Summary-CSV gefunden.")
        return []

    print(f"Lese Summary: {summary_csv}")
    df = pd.read_csv(summary_csv)
    df.columns = df.columns.str.strip()

    sim_col = 'fhe_simulate_f1_weighted'
    if sim_col not in df.columns:
        sim_col = 'pytorch_test_f1_weighted'
    
    if sim_col in df.columns:
        df[sim_col] = pd.to_numeric(df[sim_col], errors='coerce')

    tasks = []
    unique_configs = df['config_set_name'].unique()
    
    for config_name in unique_configs:
        subset = df[df['config_set_name'] == config_name]
        valid_reps = subset.dropna(subset=[sim_col])
        valid_reps = valid_reps[valid_reps[sim_col] > 0]

        if valid_reps.empty:
            continue

        best_idx = valid_reps[sim_col].idxmax()
        best_row = valid_reps.loc[best_idx]
        rep_num = best_row['repetition']
        
        rep_dir = best_row.get('results_directory')
        if not (isinstance(rep_dir, str) and os.path.isdir(rep_dir)):
            rep_dir = os.path.join(meta_folder, config_name, f"rep_{rep_num}")

        if not os.path.isdir(rep_dir):
            continue

        pths = glob.glob(os.path.join(rep_dir, "*.pth"))
        candidates = [p for p in pths if "_COMPRESSED" not in p and "_FHE_CIRCUIT" not in p]
        
        if not candidates:
            continue
            
        model_path = candidates[0]
        
        tasks.append({
            'config_set_name': config_name,
            'model_path': os.path.abspath(model_path),
            'rep_dir': os.path.abspath(rep_dir),
            'bits': int(best_row.get('quant_bits', 3)),
            'repetition': rep_num,
            'best_sim_score': best_row[sim_col],
            'n_hidden': best_row.get('n_hidden', 'N/A'),
            'unpruned_neurons': best_row.get('unpruned_neurons', 'N/A')
        })

    return tasks


def run_worker(args):
    """
    Worker mit 'Double-Lock' Isolierung (4 Kerne) und Unbuffered Output.
    Sucht automatisch die vom Sub-Skript erstellte Ergebnisdatei.
    """
    task_id, task, core_start, core_end = args
    model_path = task['model_path']
    rep_dir = task['rep_dir']
    bits = task['bits']
    config_name = task['config_set_name']

    # --- 1. OS Affinity Lock ---
    allowed_cores = list(range(core_start, core_end + 1))
    try:
        os.sched_setaffinity(0, allowed_cores)
    except Exception as e:
        print(f"   [Warnung Job {task_id}] Konnte Affinity nicht setzen: {e}")

    # Dataset Auswahl
    if "_E" in config_name or "Edge" in config_name:
        data_path = EDGE_DATA_PATH
    else:
        data_path = CIC_DATA_PATH

    if not os.path.exists(data_path):
        print(f"   [ABBRUCH Job {task_id}] Datensatz fehlt: {data_path}")
        return (task, False, "DATA_MISSING")

    model_basename = os.path.basename(model_path).replace(".pth", "")
    cluster_log_file = os.path.join(rep_dir, f"eval_fhe_{model_basename}_cluster_debug.txt")
    
    # Datei, die run_compressed_eval.py erstellt (mit _COMPRESSED)
    compressed_path = model_path.replace(".pth", "_COMPRESSED.pth")
    expected_result_file = os.path.join(rep_dir, f"eval_fhe_{model_basename}_COMPRESSED_execute.txt")

    # Alte Datei löschen, um sicher zu sein, dass wir eine neue lesen
    if os.path.exists(expected_result_file):
        os.remove(expected_result_file)

    cwd = os.getcwd()
    cmd_chain = []

    # Kompression
    if not os.path.exists(compressed_path):
        cmd_compress = (
            f"taskset -c {core_start}-{core_end} "
            f"python -u -m src.compression.compress_model "
            f"--model_path '{model_path}' "
            f"--bits {bits}"
        )
        cmd_chain.append(cmd_compress)
    
    # Evaluation
    cmd_eval = (
        f"taskset -c {core_start}-{core_end} "
        f"python -u -m src.evaluation.run_compressed_eval "
        f"--model_path '{compressed_path}' "
        f"--data_path '{data_path}' "
        f"--n_samples {SAMPLES} "
        f"--bits {bits} "
        f"--mode execute"
    )
    cmd_chain.append(cmd_eval)

    # Output in Debug-Log umleiten
    full_cmd = f"({' && '.join(cmd_chain)}) > {cluster_log_file} 2>&1"
    
    print(f"   [Start] Job {task_id}: {config_name} (Cores: {allowed_cores})")

    # --- 2. Env Vars Lock ---
    env = os.environ.copy()
    threads_str = str(CORES_PER_JOB)
    
    env["OMP_NUM_THREADS"] = threads_str
    env["MKL_NUM_THREADS"] = threads_str
    env["OPENBLAS_NUM_THREADS"] = threads_str
    env["TORCH_NUM_THREADS"] = threads_str
    env["TORCH_NUM_INTEROP_THREADS"] = "1"
    env["RAY_NUM_CPUS"] = threads_str
    env["OMP_PROC_BIND"] = "true"
    env["PYTHONPATH"] = cwd

    try:
        subprocess.run(full_cmd, shell=True, env=env, check=True)
        
        # Prüfung auf Ergebnisdatei
        if os.path.exists(expected_result_file):
            print(f"   [Fertig] Job {task_id}: {config_name}")
            task['log_path'] = expected_result_file
            task['compressed_model_path'] = compressed_path
            return (task, True, "COMPLETED")
        else:
            # Fallback Suche
            candidates = glob.glob(os.path.join(rep_dir, "*_execute.txt"))
            candidates.sort(key=os.path.getmtime, reverse=True)
            if candidates:
                print(f"   [Info] Job {task_id}: Alternative Datei gefunden.")
                task['log_path'] = candidates[0]
                task['compressed_model_path'] = compressed_path
                return (task, True, "COMPLETED")
            else:
                print(f"   [FEHLER] Job {task_id}: Keine Ergebnisdatei (*_execute.txt) gefunden!")
                task['log_path'] = cluster_log_file
                return (task, False, "NO_RESULT_FILE")
        
    except subprocess.CalledProcessError:
        print(f"   [FEHLER] Job {task_id}: {config_name} fehlgeschlagen!")
        task['log_path'] = cluster_log_file 
        return (task, False, "FAILED_EXECUTION")


def parse_eval_log(log_path):
    """
    Liest Metriken aus Log (Regex + TQDM Fallback + Sample-Time Average).
    Inklusive ANSI Cleaner.
    """
    data = {}
    if not log_path or not os.path.exists(log_path):
        return data
        
    with open(log_path, 'r', errors='replace') as f:
        content = f.read()

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    content = ansi_escape.sub('', content)

    # 1. Detaillierte Sample-Zeiten parsen (ohne Warmup)
    sample_times = []
    time_matches = re.findall(r"Sample\s+\d+:\s+([0-9\.]+)\s*s", content)
    if time_matches:
        sample_times = [float(t) for t in time_matches]
        if len(sample_times) > 1:
            avg_stable = sum(sample_times[1:]) / len(sample_times[1:])
            data["fhe_exec_time_per_sample_s"] = avg_stable
        elif len(sample_times) == 1:
            data["fhe_exec_time_per_sample_s"] = sample_times[0]

    # 2. Standard Metriken
    patterns = {
        "fallback_time": [r"Time per sample.*:\s+([0-9\.]+)", r"Execution Time.*:\s+([0-9\.]+)"],
        "fhe_exec_accuracy": [r"Accuracy.*:\s+([0-9\.]+)"],
        "fhe_exec_f1_weighted": [r"F1-Score \(weighted\).*: \s+([0-9\.]+)", r"F1 Weighted.*:\s+([0-9\.]+)"],
        "model_size_mb": [r"Model Size.*:\s+([0-9\.]+)", r"Model Size\s*\(MB\)\s*:\s*([0-9\.]+)"],
        "fhe_key_size_mb": [r"FHE Key Size.*:\s+([0-9\.]+)", r"Key Size.*:\s+([0-9\.]+)"],
        "avg_cpu_usage": [r"Avg CPU Usage.*:\s+([0-9\.]+)"],
        "tqdm_time": [r"([0-9\.]+)s/it"]
    }
    
    for key, pat_list in patterns.items():
        for pat in pat_list:
            match = re.search(pat, content)
            if match:
                val = float(match.group(1))
                if key == "fallback_time":
                    if "fhe_exec_time_per_sample_s" not in data:
                        data["fhe_exec_time_per_sample_s"] = val
                elif key == "tqdm_time":
                    if "fhe_exec_time_per_sample_s" not in data:
                        data["fhe_exec_time_per_sample_s"] = val
                else:
                    data[key] = val
                break
    return data


def collect_results(results_list, output_csv_path):
    print("\n--- SAMMLE ERGEBNISSE ---")
    final_rows = []
    
    for task, success, status in results_list:
        row = {
            "benchmark_status": status,
            "config_set_name": task['config_set_name'],
            "best_repetition": task['repetition'],
            "original_model_path": task['model_path'],
            "compressed_model_path": task.get('compressed_model_path', ''),
            "quant_bits": task['bits'],
            "n_hidden": task['n_hidden'],
            "unpruned_neurons": task['unpruned_neurons'],
            "best_sim_score": task['best_sim_score'],
        }
        
        # Leere Werte vorinitialisieren
        for k in ["fhe_exec_time_per_sample_s", "fhe_exec_accuracy", "fhe_exec_f1_weighted", 
                  "fhe_key_size_mb", "model_size_mb", "avg_cpu_usage"]:
            row[k] = ""
        
        if success and 'log_path' in task:
            metrics = parse_eval_log(task['log_path'])
            if metrics:
                row.update(metrics)
                t = metrics.get('fhe_exec_time_per_sample_s', 'N/A')
                print(f"   [OK] {task['config_set_name']}: Zeit={t}s")
            else:
                print(f"   [WARNUNG] Keine Metriken gefunden für {task['config_set_name']}")
                
        final_rows.append(row)

    if final_rows:
        df = pd.DataFrame(final_rows)
        # Schönere Sortierung
        cols_order = [
            "benchmark_status", "config_set_name", "best_repetition", 
            "fhe_exec_time_per_sample_s", "fhe_exec_accuracy", "fhe_exec_f1_weighted",
            "quant_bits", "best_sim_score", "avg_cpu_usage", "model_size_mb", "fhe_key_size_mb"
        ]
        # Vorhandene Spalten holen
        existing_cols = [c for c in cols_order if c in df.columns]
        other_cols = [c for c in df.columns if c not in existing_cols]
        df = df[existing_cols + other_cols]
        
        df.sort_values(by=["benchmark_status", "config_set_name"], inplace=True)
        
        df.to_csv(output_csv_path, index=False)
        print("=" * 60)
        print(f"ERGEBNISSE GESPEICHERT: {output_csv_path}")
        print("=" * 60)
        print(df[["config_set_name", "fhe_exec_time_per_sample_s"]].to_string())
    else:
        print("Keine Ergebnisse gesammelt.")


def main():
    # --- ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Paralleler FHE Benchmark auf Cluster")
    parser.add_argument("--offset", type=int, default=0, 
                        help="Start-Index für CPU-Kerne (z.B. 0, 40, 80...)")
    parser.add_argument("--filter", type=str, default="", 
                        help="Filter (Substring) für Configs (Optional, TARGET_CONFIGS im Code hat Vorrang)")
    args = parser.parse_args()

    # --- SETUP ---
    output_filename = f"final_benchmark_results_offset_{args.offset}.csv"
    output_csv_path = os.path.join(META_RUN_FOLDER, output_filename)

    print(f"--- STARTE PARALLELEN BENCHMARK ---")
    print(f"CPU Offset: {args.offset}")
    print(f"Output: {output_csv_path}")
    print(f"Isolierung: {CORES_PER_JOB} Cores pro Job")
    
    tasks = get_tasks_from_summary(META_RUN_FOLDER)
    
    # --- FILTER LOGIK ---
    # Priorität 1: Manuelle Liste im Code
    if TARGET_CONFIGS:
        print(f"--- FILTER AKTIV (Code): Beschränke auf {len(TARGET_CONFIGS)} Konfigurationen ---")
        tasks = [t for t in tasks if t['config_set_name'] in TARGET_CONFIGS]
    # Priorität 2: CLI Argument
    elif args.filter:
        print(f"--- FILTER AKTIV (CLI): Suche nach '{args.filter}' ---")
        tasks = [t for t in tasks if args.filter in t['config_set_name']]
    
    print(f"Tasks zu bearbeiten: {len(tasks)}")
    if not tasks:
        print("Nichts zu tun.")
        return

    num_processes = min(len(tasks), MAX_PARALLEL_JOBS)
    
    worker_args = []
    # Berechnung der Kerne mit Offset
    for i, task in enumerate(tasks):
        job_base_index = i * CORES_PER_JOB
        core_start = job_base_index + args.offset
        core_end = core_start + CORES_PER_JOB - 1
        
        worker_args.append((i, task, core_start, core_end))

    print(f"Starte Pool mit {num_processes} Prozessen...")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_worker, worker_args)

    collect_results(results, output_csv_path)


if __name__ == "__main__":
    main()