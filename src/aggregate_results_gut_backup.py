import os
import glob
import pandas as pd
import argparse
import re
import numpy as np

def parse_txt_metrics(file_path):
    """Parst Metriken aus der Textdatei (Simulate/Execute Log)."""
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Regex für Header-Metriken
        # Suche nach "F1(w): 0.1234"
        f1_match = re.search(r"F1\(w\):\s+([0-9\.]+)", content)
        if f1_match: metrics['fhe_f1_weighted'] = float(f1_match.group(1))
        
        roc_match = re.search(r"ROC-AUC\(m\):\s+([0-9\.]+)", content)
        if roc_match: metrics['fhe_roc_auc_macro'] = float(roc_match.group(1))
        
        # Regex für Tabelle "weighted avg"
        # Format: weighted avg       0.81      0.90      0.85        10
        avg_match = re.search(r"weighted avg\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)", content)
        if avg_match:
            metrics['fhe_precision_weighted'] = float(avg_match.group(1))
            metrics['fhe_recall_weighted'] = float(avg_match.group(2))
            
    except Exception as e:
        # print(f"Fehler beim Lesen von {file_path}: {e}")
        pass
    return metrics

def extract_config_params(folder_name):
    """Versucht, Parameter (Bits, Pruning) aus dem Ordnernamen zu erraten."""
    params = {'config_name': folder_name}
    
    # Bits (qbX)
    bits_match = re.search(r'qb(\d+)', folder_name)
    if bits_match: params['quant_bits'] = int(bits_match.group(1))
    else: params['quant_bits'] = 0
    
    # Pruning (pXX)
    prune_match = re.search(r'_p(\d+)_', folder_name)
    if prune_match: params['unpruned_neurons'] = int(prune_match.group(1))
    else: params['unpruned_neurons'] = 0
    
    # Dataset (Endung _C oder _E)
    if folder_name.endswith("_C"): params['dataset'] = "CICIoT"
    elif folder_name.endswith("_E"): params['dataset'] = "EgeIIoT"
    else: params['dataset'] = "Unknown"
    
    # Strategy
    if "unstructured" in folder_name.lower(): params['strategy'] = "Unstructured"
    elif "structured" in folder_name.lower(): params['strategy'] = "Structured"
    else: params['strategy'] = "Unknown"
    
    return params

def aggregate_folder(meta_run_path):
    if not os.path.exists(meta_run_path):
        print(f"Fehler: Pfad '{meta_run_path}' existiert nicht.")
        return

    print(f"Starte Aggregation für Meta-Run: {meta_run_path}")
    
    # Liste für alle gefundenen Einzel-Runs (Repetitions)
    all_runs_data = []
    
    # 1. Iteriere über Konfigurations-Ordner (direkte Unterordner des Meta-Runs)
    # Wir ignorieren Ordner wie 'run_configs'
    config_dirs = [d for d in os.listdir(meta_run_path) if os.path.isdir(os.path.join(meta_run_path, d))]
    
    print(f"Gefundene Konfigurations-Ordner: {len(config_dirs)}")

    for conf_dir_name in config_dirs:
        conf_path = os.path.join(meta_run_path, conf_dir_name)
        
        # Parameter aus dem Namen extrahieren (für Sortierung/Gruppierung)
        config_params = extract_config_params(conf_dir_name)
        
        # 2. Iteriere über Repetitions (rep_*)
        rep_dirs = glob.glob(os.path.join(conf_path, "rep_*"))
        
        if not rep_dirs:
            continue # Kein rep Ordner, wahrscheinlich ein Config-Ordner
            
        for rep_dir in rep_dirs:
            rep_name = os.path.basename(rep_dir)
            
            # Metriken suchen
            metrics = {}
            
            # Suche nach TXT Datei (Simulate hat Vorrang, sonst Execute)
            txt_files = glob.glob(os.path.join(rep_dir, "eval_fhe_*simulate*.txt"))
            if not txt_files:
                txt_files = glob.glob(os.path.join(rep_dir, "eval_fhe_*.txt"))
            
            if txt_files:
                # Nimm die neueste Textdatei
                target_txt = max(txt_files, key=os.path.getmtime)
                metrics = parse_txt_metrics(target_txt)
            
            # Wenn wir Metriken gefunden haben, speichern wir den Run
            if metrics:
                run_data = config_params.copy()
                run_data['repetition'] = rep_name
                run_data.update(metrics)
                all_runs_data.append(run_data)

    # --- DATENVERARBEITUNG ---
    if not all_runs_data:
        print("Keine gültigen Ergebnis-Dateien (.txt) in den Rep-Ordnern gefunden.")
        return

    df = pd.DataFrame(all_runs_data)
    print(f"Verarbeite {len(df)} erfolgreiche Runs.")

    # Metrik-Spalten identifizieren (alle die mit 'fhe_' beginnen)
    metric_cols = [c for c in df.columns if c.startswith('fhe_')]
    
    # Gruppieren nach Konfiguration
    group_cols = ['config_name', 'dataset', 'strategy', 'quant_bits', 'unpruned_neurons']
    
    # Aggregation (Mean + Std)
    agg_df = df.groupby(group_cols)[metric_cols].agg(['mean', 'std', 'count']).reset_index()
    
    # Spaltennamen flachklopfen
    agg_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_df.columns.values]
    
    # Aufräumen der Count Spalten (wir brauchen nur eine)
    count_cols = [c for c in agg_df.columns if c.endswith('_count')]
    if count_cols:
        agg_df['total_runs'] = agg_df[count_cols[0]]
        agg_df = agg_df.drop(columns=count_cols)

    # Sortierung
    agg_df = agg_df.sort_values(by=['dataset', 'strategy', 'unpruned_neurons', 'quant_bits'])

    # --- RUNDUNG AUF 3 NACHKOMMASTELLEN ---
    agg_df = agg_df.round(3)
    # --------------------------------------

    # Speichern im Meta-Run Ordner
    output_file = os.path.join(meta_run_path, "AGGREGATED_RESULTS.csv")
    agg_df.to_csv(output_file, index=False)
    
    print(f"\nErfolg! Aggregierte Datei gespeichert unter:\n{output_file}")
    
    # Vorschau
    print("\nVorschau:")
    print(agg_df[['config_name', 'total_runs', 'fhe_f1_weighted_mean']].to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregiert Ergebnisse aus einem spezifischen Meta-Run Ordner.")
    parser.add_argument("folder", help="Pfad zum Meta-Run Ordner (z.B. results/qat_runs/meta_run_2025...)")
    args = parser.parse_args()
    
    aggregate_folder(args.folder)