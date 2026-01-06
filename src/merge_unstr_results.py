import os
import glob
import pandas as pd
import re

# --- KONFIGURATION ---
SOURCE_DIR = "results/qat_runs/meta_results_unstr_new"
FILE_PATTERN = "meta_run_*_q357_p*_new_unstr.csv"
OUTPUT_FILE = "FINAL_MERGED_RESULTS_UNSTR_357.csv"

def extract_numeric_pruning(config_name):
    match = re.search(r'p(\d+)', str(config_name))
    return int(match.group(1)) if match else 0

def merge_csvs():
    search_path = os.path.join(SOURCE_DIR, FILE_PATTERN)
    print(f"Suche nach: {search_path}")
    
    all_files = glob.glob(search_path)
    if not all_files:
        print("FEHLER: Keine Dateien gefunden!")
        return

    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            print(f"  -> Fehler bei {filename}: {e}")

    if not df_list: return

    full_df = pd.concat(df_list, ignore_index=True)
    
    # --- SORTIERUNG ---
    
    # 1. Dataset Namen vereinheitlichen/sortierbar machen
    if 'dataset' in full_df.columns:
        dataset_col = 'dataset'
    elif 'dataset_name_from_log' in full_df.columns:
        dataset_col = 'dataset_name_from_log'
    else:
        dataset_col = None

    # 2. Quant Bits
    if 'quant_bits' in full_df.columns:
        full_df['quant_bits'] = pd.to_numeric(full_df['quant_bits'], errors='coerce')
    
    # 3. Pruning (aus Spalte oder Name extrahieren)
    if 'unpruned_neurons' in full_df.columns:
        full_df['sort_prun'] = pd.to_numeric(full_df['unpruned_neurons'], errors='coerce')
    elif 'config_name' in full_df.columns:
        full_df['sort_prun'] = full_df['config_name'].apply(extract_numeric_pruning)
    elif 'config_set_name' in full_df.columns:
        full_df['sort_prun'] = full_df['config_set_name'].apply(extract_numeric_pruning)
    else:
        full_df['sort_prun'] = 0

    # Sortier-Liste erstellen
    sort_by = []
    if dataset_col: sort_by.append(dataset_col) # 1. Dataset
    if 'quant_bits' in full_df.columns: sort_by.append('quant_bits') # 2. Quantisierung
    sort_by.append('sort_prun') # 3. Pruning
    
    print(f"Sortiere nach: {sort_by}")
    
    full_df = full_df.sort_values(by=sort_by)
    
    # Hilfsspalte weg
    full_df = full_df.drop(columns=['sort_prun'])

    # Speichern
    full_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Fertig! Gespeichert in: {OUTPUT_FILE}")
    
    # Vorschau
    preview_cols = [c for c in [dataset_col, 'quant_bits', 'unpruned_neurons'] if c in full_df.columns]
    print(full_df[preview_cols].to_string(index=False))

if __name__ == "__main__":
    merge_csvs()