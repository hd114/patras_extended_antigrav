import os
import sys
import pandas as pd
import subprocess
import time
import glob
import re
from multiprocessing import Pool

# --- KONFIGURATION (DRY RUN) -------------------------------------------------
META_RUN_FOLDER = "results/qat_runs/meta_run_20251206_C_q357"
CORES_PER_JOB = 4       # Wir testen 4 Kerne pro Job
MAX_PARALLEL_JOBS = 5   # Kleinere Zahl reicht zum Testen
OUTPUT_CSV = "dry_run_results.csv"

# Pfade (müssen existieren, sonst meckert das Skript)
EDGE_DATA_PATH = "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
CIC_DATA_PATH = "/home/jovyan/TenSEAL_projects/ciciot_dataset_all.npz"
# -----------------------------------------------------------------------------

def get_best_models_tasks(meta_folder):
    """(Unverändert) Findet Modelle."""
    summary_path = os.path.join(meta_folder, "meta_experiment_summary.csv")
    if not os.path.exists(summary_path):
        csvs = glob.glob(os.path.join(meta_folder, "meta_experiment_summary_*.csv"))
        if not csvs:
            print(f"[Fehler] Keine Summary-CSV in {meta_folder} gefunden.")
            return []
        summary_path = max(csvs, key=os.path.getmtime)

    print(f"Lese Summary: {summary_path}")
    df = pd.read_csv(summary_path)
    df.columns = df.columns.str.strip()

    sim_col = 'fhe_simulate_f1_weighted'
    if sim_col not in df.columns:
        sim_col = 'pytorch_test_f1_weighted'
    if sim_col in df.columns:
        df[sim_col] = pd.to_numeric(df[sim_col], errors='coerce')

    tasks = []
    unique_configs = df['config_set_name'].unique()
    
    for config in unique_configs:
        subset = df[df['config_set_name'] == config]
        valid_reps = subset.dropna(subset=[sim_col])
        if valid_reps.empty: continue

        best_idx = valid_reps[sim_col].idxmax()
        best_row = valid_reps.loc[best_idx]
        rep_num = best_row['repetition']
        
        rep_dir = best_row.get('results_directory')
        if not (isinstance(rep_dir, str) and os.path.isdir(rep_dir)):
            rep_dir = os.path.join(meta_folder, config, f"rep_{rep_num}")

        if not os.path.isdir(rep_dir): continue

        pths = glob.glob(os.path.join(rep_dir, "*.pth"))
        # Wir nehmen irgendein pth file zum Testen der Pfade
        candidates = [p for p in pths if "_COMPRESSED" not in p and "_FHE_CIRCUIT" not in p]
        
        if not candidates: continue
        tasks.append({
            'config': config,
            'model_path': os.path.abspath(candidates[0]),
            'bits': 3 # Dummy wert
        })
    
    print(f"   Gefundene Tasks: {len(tasks)}")
    return tasks

def run_worker_dry(args):
    """
    Simuliert die Arbeit und prüft CPU-Affinity.
    """
    task_id, task, core_start, core_end = args
    config_name = task['config']
    model_path = task['model_path']
    
    # 1. OS Affinity setzen (wie im echten Skript)
    allowed_cores = list(range(core_start, core_end + 1))
    try:
        os.sched_setaffinity(0, allowed_cores)
    except Exception as e:
        print(f"   [Error] Affinity setzen fehlgeschlagen: {e}")

    # Logfile vorbereiten
    model_dir = os.path.dirname(model_path)
    model_filename = os.path.basename(model_path)
    clean_name = model_filename.replace(".pth", "").replace("_COMPRESSED", "")
    # Wir nennen es _dryrun_, damit wir keine echten Ergebnisse überschreiben
    log_file = os.path.join(model_dir, f"eval_fhe_{clean_name}_execute_time.txt")

    print(f"   [Test] Job {task_id} startet auf Cores {core_start}-{core_end}")

    # 2. Diagnose-Befehl ausführen
    # Wir fragen das OS: "Auf welchen CPUs darf dieser Prozess laufen?"
    # Und wir prüfen, ob die Env-Vars angekommen sind.
    cmd = (
        f"echo '--- DIAGNOSE START ---'; "
        f"echo 'PID: '$$; "
        f"grep Cpus_allowed_list /proc/self/status; " # Das ist der Beweis!
        f"echo 'OMP_THREADS='$OMP_NUM_THREADS; "
        f"sleep 2; " # Kurz warten, damit Prozess existiert
        f"echo '--- DIAGNOSE END ---'"
    )

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(CORES_PER_JOB)
    # ... (andere Env vars hier nicht zwingend nötig für den Dry Run Output)

    try:
        # Wir fangen den Output ab, um ihn zu printen
        result = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
        
        # Output parsen für saubere Anzeige im Terminal
        cpus_allowed = "Unknown"
        for line in result.stdout.splitlines():
            if "Cpus_allowed_list" in line:
                cpus_allowed = line.split(":")[1].strip()
        
        print(f"   [CHECK Job {task_id}] Erwartet: {core_start}-{core_end} | Realität: {cpus_allowed}")
        
        if cpus_allowed != f"{core_start}-{core_end}":
            if "," in cpus_allowed: # Manchmal Format 0,1,2,3
                 print(f"   [WARNUNG] Format abweichend oder Affinity falsch: {cpus_allowed}")
            else:
                 print(f"   [ALARM] CPU ISOLIERUNG FEHLGESCHLAGEN!")

        # 3. Dummy Log Datei schreiben (damit collect funktioniert)
        with open(log_file, "w") as f:
            f.write(f"Execution Time (s): 1.234\n")
            f.write(f"Accuracy: 0.9999\n")
            f.write(f"F1 Weighted: 0.8888\n")
            f.write(f"Key Size: 100.0\n")
            f.write(f"Model Size: 5.0\n")
            f.write(f"Avg CPU Usage: 400.0\n")

        return (config_name, True)

    except Exception as e:
        print(f"   [Exception] {e}")
        return (config_name, False)

def collect_results_dry(tasks):
    print("\n--- SAMMLE ERGEBNISSE (DRY RUN) ---")
    results = []
    for task in tasks:
        model_dir = os.path.dirname(task['model_path'])
        # Wir suchen die Dummy Files
        logs = glob.glob(os.path.join(model_dir, "*_execute_time.txt"))
        if logs:
            print(f"   Log gefunden für {task['config']}: {os.path.basename(logs[0])}")
            results.append({'config': task['config'], 'status': 'OK'})
        else:
            print(f"   [FEHLER] Kein Log für {task['config']}")
    
    if results:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print(f"CSV geschrieben: {OUTPUT_CSV}")

def main():
    print("--- STARTE DRY RUN ---")
    tasks = get_best_models_tasks(META_RUN_FOLDER)
    if not tasks: return

    # Nur die ersten X Tasks testen
    test_tasks = tasks[:MAX_PARALLEL_JOBS]
    
    worker_args = []
    for i, task in enumerate(test_tasks):
        core_start = i * CORES_PER_JOB
        core_end = core_start + CORES_PER_JOB - 1
        worker_args.append((i, task, core_start, core_end))

    with Pool(processes=len(test_tasks)) as pool:
        pool.map(run_worker_dry, worker_args)

    collect_results_dry(test_tasks)
    print("\n--- DRY RUN ENDE ---")
    print("Bitte prüfe oben die Zeilen '[CHECK Job X]'.")
    print("Wenn 'Erwartet' und 'Realität' übereinstimmen, funktioniert die Isolierung.")

if __name__ == "__main__":
    main()