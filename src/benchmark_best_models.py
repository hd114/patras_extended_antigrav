import os
import glob
import pandas as pd
import argparse
import subprocess
import sys
import re
import numpy as np

def find_summary_csv(meta_folder):
    """Findet die neueste Summary CSV im Meta-Run Ordner."""
    csvs = glob.glob(os.path.join(meta_folder, "meta_experiment_summary_*.csv"))
    if not csvs:
        return None
    return max(csvs, key=os.path.getmtime)

def find_model_pth(results_dir):
    """Sucht das originale .pth Modell im Ergebnisordner."""
    if not os.path.isdir(results_dir): return None
    pths = glob.glob(os.path.join(results_dir, "*.pth"))
    candidates = [p for p in pths if "_COMPRESSED" not in p and "_FHE_CIRCUIT" not in p]
    if candidates:
        return candidates[0]
    return None

def parse_eval_log(log_path):
    """Liest Metriken aus dem eval_fhe_..._execute.txt Log aus."""
    data = {}
    if not os.path.exists(log_path):
        return data
        
    with open(log_path, 'r') as f:
        content = f.read()
        
    patterns = {
        "fhe_time_per_sample_s": r"Time per sample:\s+([0-9\.]+)",
        "fhe_accuracy": r"Accuracy:\s+([0-9\.]+)",
        "fhe_f1_weighted": r"F1-Score \(weighted\):\s+([0-9\.]+)",
        "model_size_mb": r"Model Size\s+:\s+([0-9\.]+)",
        "key_size_mb": r"FHE Key Size\s+:\s+([0-9\.]+)",
        "avg_cpu_usage": r"Avg CPU Usage:\s+([0-9\.]+)"
    }
    
    for key, pat in patterns.items():
        match = re.search(pat, content)
        if match:
            try:
                data[key] = float(match.group(1))
            except ValueError:
                data[key] = -1.0
            
    return data

def main(meta_folder, n_samples):
    print(f"--- Starte Benchmark für Best-Of-Reps in: {meta_folder} ---")
    
    summary_csv = find_summary_csv(meta_folder)
    if not summary_csv:
        print("FEHLER: Keine 'meta_experiment_summary_*.csv' gefunden.")
        return

    print(f"Lade Summary: {summary_csv}")
    df = pd.read_csv(summary_csv)
    
    # Spalten säubern
    df.columns = df.columns.str.strip()
    
    # Prüfe auf Simulations-Score
    sim_col = 'fhe_simulate_f1_weighted'
    if sim_col not in df.columns:
        print(f"WARNUNG: '{sim_col}' nicht gefunden. Versuche 'pytorch_test_f1_weighted' als Fallback.")
        sim_col = 'pytorch_test_f1_weighted'
        
    if sim_col in df.columns:
        df[sim_col] = pd.to_numeric(df[sim_col], errors='coerce')
    
    # Eindeutige Konfigurationen identifizieren
    unique_configs = df['config_set_name'].unique()
    print(f"Analysiere {len(unique_configs)} Konfigurationen...")
    
    final_results = []
    
    for config_name in unique_configs:
        print(f"\nPrüfe Konfiguration: {config_name}")
        
        # Alle Repetitions für diese Config holen
        subset = df[df['config_set_name'] == config_name]
        
        # Validierung: Gibt es überhaupt einen gültigen Score?
        valid_reps = subset.dropna(subset=[sim_col])
        valid_reps = valid_reps[valid_reps[sim_col] > 0] # Optional: Filter für 0-Scores
        
        if valid_reps.empty:
            print(f"  [SKIP] Keine validen FHE Scores in {len(subset)} Reps gefunden.")
            
            # Eintrag für CSV erstellen (damit man weiß, dass es gefehlt hat)
            skipped_entry = subset.iloc[0].to_dict() # Metadaten von irgendeiner Rep nehmen
            result_row = {
                "benchmark_status": "SKIPPED_NO_SCORE",
                "config_set_name": config_name,
                "quant_bits": skipped_entry.get('quant_bits', 'N/A'),
                "n_hidden": skipped_entry.get('n_hidden', 'N/A'),
                "unpruned_neurons": skipped_entry.get('unpruned_neurons', 'N/A'),
                "best_sim_score": "N/A"
            }
            final_results.append(result_row)
            continue
            
        # Beste Repetition auswählen
        best_idx = valid_reps[sim_col].idxmax()
        best_row = valid_reps.loc[best_idx]
        best_score = best_row[sim_col]
        rep_num = best_row['repetition']
        
        print(f"  -> Beste Repetition: {rep_num} (Score: {best_score:.4f})")
        
        # Pfad zum Ergebnisordner bestimmen
        rep_dir = best_row.get('results_directory')
        
        # Fallback, falls der Pfad in der CSV absolut und falsch ist (z.B. anderer Server)
        # Wir bauen ihn relativ zum meta_folder neu auf
        if not (rep_dir and os.path.isdir(rep_dir)):
            rep_dir = os.path.join(meta_folder, config_name, f"rep_{rep_num}")
            
        if not os.path.isdir(rep_dir):
            print(f"  [ERROR] Ordner nicht gefunden: {rep_dir}")
            result_row = best_row.to_dict()
            result_row.update({"benchmark_status": "ERROR_DIR_NOT_FOUND"})
            final_results.append(result_row)
            continue

        model_path = find_model_pth(rep_dir)
        if not model_path:
            print(f"  [ERROR] Kein .pth Modell in {rep_dir}")
            result_row = best_row.to_dict()
            result_row.update({"benchmark_status": "ERROR_NO_PTH"})
            final_results.append(result_row)
            continue

        # --- EXECUTION ---
        compressed_path = model_path.replace(".pth", "_COMPRESSED.pth")
        
        # 1. Kompression (falls nötig)
        if not os.path.exists(compressed_path):
            print("  Starte Kompression...")
            bits = int(best_row.get('quant_bits', 3))
            try:
                subprocess.check_call([sys.executable, "-m", "src.compression.compress_model", 
                                       "--model_path", model_path, "--bits", str(bits)])
            except subprocess.CalledProcessError:
                print("  [FAIL] Kompression fehlgeschlagen.")
                result_row = best_row.to_dict()
                result_row.update({"benchmark_status": "FAILED_COMPRESSION"})
                final_results.append(result_row)
                continue
        
        # 2. Evaluation
        model_basename = os.path.basename(compressed_path).replace(".pth", "")
        eval_log = os.path.join(rep_dir, f"eval_fhe_{model_basename}_execute.txt")
        
        # Immer ausführen (oder hier Logik ändern, falls man existierende überspringen will)
        print(f"  Starte Execute Eval ({n_samples} Samples)...")
        if "_E" in config_name or "Edge" in config_name:
            data_path = "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
        else:
            data_path = "/home/jovyan/TenSEAL_projects/ciciot_dataset_all.npz"
            
        try:
            bits = int(best_row.get('quant_bits', 3))
            cmd = [sys.executable, "-m", "src.evaluation.run_compressed_eval",
                   "--model_path", compressed_path,
                   "--data_path", data_path,
                   "--bits", str(bits),
                   "--mode", "execute",
                   "--n_samples", str(n_samples)]
            
            # Wir leiten stdout in eine Logdatei im Rep-Ordner um, damit wir ALLES haben
            process_log_path = os.path.join(rep_dir, "benchmark_process_log.txt")
            with open(process_log_path, "w") as f_log:
                subprocess.check_call(cmd, stdout=f_log, stderr=subprocess.STDOUT)
                
            print("  [OK] Evaluation abgeschlossen.")
            status = "COMPLETED"
            
        except subprocess.CalledProcessError:
            print("  [FAIL] Evaluation fehlgeschlagen (siehe benchmark_process_log.txt).")
            status = "FAILED_EXECUTION"
            
        # 3. Ergebnisse sammeln
        metrics = parse_eval_log(eval_log)
        
        # Basis-Infos
        result_row = {
            "benchmark_status": status,
            "config_set_name": config_name,
            "best_repetition": rep_num,
            "original_model_path": model_path,
            "compressed_model_path": compressed_path,
            "quant_bits": best_row.get('quant_bits', 'N/A'),
            "n_hidden": best_row.get('n_hidden', 'N/A'),
            "unpruned_neurons": best_row.get('unpruned_neurons', 'N/A'),
            "best_sim_score": best_score,
            
            # Gelesene Metriken
            "fhe_exec_time_per_sample_s": metrics.get("fhe_time_per_sample_s", ""),
            "fhe_exec_accuracy": metrics.get("fhe_accuracy", ""),
            "fhe_exec_f1_weighted": metrics.get("fhe_f1_weighted", ""),
            "fhe_key_size_mb": metrics.get("key_size_mb", ""),
            "model_size_mb": metrics.get("model_size_mb", ""),
            "avg_cpu_usage": metrics.get("avg_cpu_usage", "")
        }
        final_results.append(result_row)

    # Final Save
    if final_results:
        out_df = pd.DataFrame(final_results)
        # Sortieren für bessere Übersicht
        out_df.sort_values(by=["benchmark_status", "config_set_name"], inplace=True)
        
        out_path = os.path.join(meta_folder, "final_fhe_benchmark_best_reps.csv")
        out_df.to_csv(out_path, index=False)
        print("\n" + "="*60)
        print(f"Benchmark abgeschlossen! Ergebnisse gespeichert in:\n{out_path}")
        print("="*60)
        # Kurze Übersicht ausgeben
        print(out_df[["config_set_name", "benchmark_status", "fhe_exec_time_per_sample_s"]].to_string())
    else:
        print("Keine Ergebnisse gesammelt.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_folder", help="Pfad zum Meta-Run Ordner")
    parser.add_argument("--samples", type=int, default=10, help="Anzahl Samples für Execute")
    args = parser.parse_args()
    
    main(args.meta_folder, args.samples)