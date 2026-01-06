import os
import glob
import pandas as pd
import argparse
import re
import numpy as np

def parse_txt_metrics(file_path):
    """Parst Metriken aus der Textdatei. Holt Macro Prec/Rec und Weighted F1."""
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # --- 1. F1 Weighted (Bleibt weighted) ---
        f1_match = re.search(r"F1\(w\):\s+([0-9\.]+)", content)
        if f1_match: metrics['fhe_f1_weighted'] = float(f1_match.group(1))
        
        # --- 2. ROC AUC Macro (Bleibt macro) ---
        roc_match = re.search(r"ROC-AUC\(m\):\s+([0-9\.]+)", content)
        if roc_match: metrics['fhe_roc_auc_macro'] = float(roc_match.group(1))
        
        # --- 3. Precision Macro (Neu: Prec(m)) ---
        prec_match = re.search(r"Prec\(m\):\s+([0-9\.]+)", content)
        if prec_match: metrics['fhe_precision_macro'] = float(prec_match.group(1))
        
        # --- 4. Recall Macro (Neu: Rec(m)) ---
        rec_match = re.search(r"Rec\(m\):\s+([0-9\.]+)", content)
        if rec_match: metrics['fhe_recall_macro'] = float(rec_match.group(1))
            
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

def process_single_meta_run(meta_run_path):
    """Verarbeitet einen einzelnen Meta-Run Ordner."""
    if not os.path.isdir(meta_run_path):
        return

    print(f"--> Analysiere Ordner: {meta_run_path}")
    
    # Liste für alle gefundenen Einzel-Runs (Repetitions)
    all_runs_data = []
    
    # 1. Iteriere über Konfigurations-Ordner
    config_dirs = [d for d in os.listdir(meta_run_path) if os.path.isdir(os.path.join(meta_run_path, d))]
    # Filter: Wir wollen nur echte Config-Ordner (die mit qb beginnen)
    config_dirs = [d for d in config_dirs if d.startswith("qb")]

    for conf_dir_name in config_dirs:
        conf_path = os.path.join(meta_run_path, conf_dir_name)
        
        # Parameter extrahieren
        config_params = extract_config_params(conf_dir_name)
        
        # 2. Iteriere über Repetitions (rep_*)
        rep_dirs = glob.glob(os.path.join(conf_path, "rep_*"))
        
        for rep_dir in rep_dirs:
            rep_name = os.path.basename(rep_dir)
            
            # Suche nach TXT Datei (Simulate hat Vorrang)
            txt_files = glob.glob(os.path.join(rep_dir, "eval_fhe_*simulate*.txt"))
            if not txt_files:
                txt_files = glob.glob(os.path.join(rep_dir, "eval_fhe_*.txt"))
            
            metrics = {}
            if txt_files:
                # Nimm die neueste Textdatei
                target_txt = max(txt_files, key=os.path.getmtime)
                metrics = parse_txt_metrics(target_txt)
            
            # Daten zusammenführen
            run_data = config_params.copy()
            run_data['repetition'] = rep_name
            run_data.update(metrics)
            all_runs_data.append(run_data)

    if not all_runs_data:
        print("   Keine Daten gefunden.")
        return

    df = pd.DataFrame(all_runs_data)
    
    # Metrik-Spalten identifizieren (alle die mit 'fhe_' beginnen)
    metric_cols = [c for c in df.columns if c.startswith('fhe_')]
    
    # Gruppieren nach Konfiguration
    group_cols = ['config_name', 'dataset', 'strategy', 'quant_bits', 'unpruned_neurons']
    
    # Aggregation (Mean + Std)
    agg_df = df.groupby(group_cols)[metric_cols].agg(['mean', 'std', 'count']).reset_index()
    
    # Spaltennamen flachklopfen
    agg_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in agg_df.columns.values]
    
    # Aufräumen der Count Spalten
    count_cols = [c for c in agg_df.columns if c.endswith('_count')]
    if count_cols:
        agg_df['total_runs'] = agg_df[count_cols[0]]
        agg_df = agg_df.drop(columns=count_cols)

    # Sortierung
    agg_df = agg_df.sort_values(by=['dataset', 'strategy', 'unpruned_neurons', 'quant_bits'])
    
    # Runden auf 3 Stellen
    agg_df = agg_df.round(3)

    # Speichern im Meta-Run Ordner
    output_file = os.path.join(meta_run_path, "AGGREGATED_RESULTS.csv")
    agg_df.to_csv(output_file, index=False)
    
    print(f"   Erfolg! Gespeichert in: {output_file}")
    
    # Vorschau
    preview_cols = [c for c in agg_df.columns if 'fhe_' in c and 'mean' in c]
    if preview_cols:
        print(agg_df[['config_name'] + preview_cols].head().to_string(index=False))

def main(input_pattern):
    base_search_path = "results/qat_runs"
    
    # --- INTELLIGENTE PFAD-FINDUNG ---
    direct_matches = glob.glob(input_pattern)
    relative_pattern = os.path.join(base_search_path, input_pattern)
    relative_matches = glob.glob(relative_pattern)
    
    final_matches = []
    if direct_matches: final_matches = direct_matches
    elif relative_matches: final_matches = relative_matches
    else:
        if not input_pattern.endswith("*"):
             extended_pattern = os.path.join(base_search_path, input_pattern + "*")
             final_matches = glob.glob(extended_pattern)

    if not final_matches:
        print(f"FEHLER: Konnte keine Ordner finden, die auf '{input_pattern}' passen.")
        return

    print(f"Gefunden: {len(final_matches)} Meta-Run Ordner.")
    
    for folder in final_matches:
        if os.path.isdir(folder):
            process_single_meta_run(folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Pfad oder Pattern zum Meta-Run Ordner")
    args = parser.parse_args()
    
    main(args.folder)