import os
import glob
import re
import pandas as pd

# --- KONFIGURATION ---
# Dein Meta-Run Ordner
META_RUN_FOLDER = "results/qat_runs/meta_run_20251207_E_q_357_pall_new_str"
OUTPUT_CSV = "final_recovered_results.csv"
# ---------------------

def parse_eval_file(filepath):
    """Liest die Metriken aus einer eval_fhe_*_execute.txt Datei."""
    data = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Regex Muster für die Metriken
        patterns = {
            "fhe_time_per_sample_s": [r"Time per sample.*:\s+([0-9\.]+)", r"Execution Time.*:\s+([0-9\.]+)"],
            "fhe_accuracy": [r"Accuracy.*:\s+([0-9\.]+)"],
            "fhe_f1_weighted": [r"F1 Weighted.*:\s+([0-9\.]+)", r"F1-Score \(weighted\).*: \s+([0-9\.]+)"],
            "key_size_mb": [r"Key Size.*:\s+([0-9\.]+)", r"FHE Key Size.*:\s+([0-9\.]+)"],
            "model_size_mb": [r"Model Size.*:\s+([0-9\.]+)"],
            "avg_cpu_usage": [r"Avg CPU Usage.*:\s+([0-9\.]+)"]
        }
        
        for key, pat_list in patterns.items():
            for pat in pat_list:
                match = re.search(pat, content)
                if match:
                    data[key] = float(match.group(1))
                    break # Ersten Treffer nehmen
    except Exception as e:
        print(f"Fehler beim Lesen von {filepath}: {e}")
    return data

def main():
    print(f"Suche nach Ergebnissen in: {META_RUN_FOLDER}")
    
    # Suche rekursiv nach allen Dateien, die auf _execute.txt enden
    # Pattern: META_RUN_FOLDER/**/eval_fhe_*_execute.txt
    search_pattern = os.path.join(META_RUN_FOLDER, "**", "eval_fhe_*_execute.txt")
    found_files = glob.glob(search_pattern, recursive=True)
    
    # Filter: "simulate" Dateien ignorieren, falls welche da sind
    valid_files = [f for f in found_files if "simulate" not in os.path.basename(f)]
    
    print(f"Gefundene Ergebnis-Dateien: {len(valid_files)}")
    
    results = []
    
    for filepath in valid_files:
        print(f" -> Lese: {os.path.basename(filepath)}")
        metrics = parse_eval_file(filepath)
        
        # Versuchen, Config-Namen aus dem Pfad oder Dateinamen abzuleiten
        # Dateiname ist z.B. eval_fhe_QAT_EgeIIoT_100h_qb3_ep96_f1w9281_execute.txt
        # Ordnerstruktur: .../qb3_p8_structured_E/rep_3/...
        
        # Wir holen den Config-Namen aus dem Eltern-Eltern-Ordner
        # filepath = .../qb3_p8_structured_E/rep_3/eval...txt
        parent = os.path.dirname(filepath) # rep_3
        grandparent = os.path.dirname(parent) # qb3_p8_structured_E
        config_name = os.path.basename(grandparent)
        
        # Wir holen die Bits aus dem Config-Namen (z.B. qb3 -> 3)
        bits_match = re.search(r"qb(\d+)", config_name)
        bits = int(bits_match.group(1)) if bits_match else 0
        
        row = {
            "config_set_name": config_name,
            "bits": bits,
            "log_path": filepath
        }
        row.update(metrics)
        
        # Nur aufnehmen, wenn wir wirklich Daten haben
        if "fhe_time_per_sample_s" in metrics:
            results.append(row)
        else:
            print(f"    [WARNUNG] Datei leer oder unvollständig.")

    if results:
        df = pd.DataFrame(results)
        # Sortieren
        df.sort_values(by="config_set_name", inplace=True)
        
        # Speichern im aktuellen Ordner
        df.to_csv(OUTPUT_CSV, index=False)
        print("\n" + "="*60)
        print(f"RETTUNG ERFOLGREICH! Datei gespeichert als: {OUTPUT_CSV}")
        print("="*60)
        print(df[["config_set_name", "fhe_time_per_sample_s", "fhe_f1_weighted"]].to_string())
    else:
        print("\n[FEHLER] Keine gültigen Daten extrahiert. Sind die execute.txt Dateien wirklich da?")

if __name__ == "__main__":
    main()