import pandas as pd
import numpy as np

def compare_csvs(file1, file2):
    print(f"Vergleiche:\n1: {file1}\n2: {file2}\n")
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Gemeinsame Spalten für den Join (Schlüssel)
    # Wir müssen 'config_set_name' in df1 mit 'config_name' in df2 matchen
    # oder wir nutzen dataset, bits, pruning als Schlüssel
    
    # Umbenennen für Join
    if 'config_name' in df2.columns:
        df2 = df2.rename(columns={'config_name': 'config_set_name'})
        
    # Schlüsselspalten
    join_cols = ['config_set_name']
    
    # Merge
    merged = pd.merge(df1, df2, on=join_cols, suffixes=('_csv', '_txt'), how='inner')
    
    print(f"Gefundene gemeinsame Zeilen: {len(merged)}")
    
    # Vergleich der F1-Werte
    # Spaltennamen anpassen an deine CSVs
    col1 = 'fhe_simulate_f1_weighted_mean' # aus csv basierter datei
    col2 = 'fhe_f1_weighted_mean'          # aus txt basierter datei
    
    print(f"\nVergleich {col1} (CSV-Quelle) vs {col2} (TXT-Quelle):")
    print("-" * 60)
    print(f"{'Config':<30} | {'CSV-Wert':<10} | {'TXT-Wert':<10} | {'Diff':<10}")
    print("-" * 60)
    
    for index, row in merged.iterrows():
        val1 = row.get(col1, np.nan)
        val2 = row.get(col2, np.nan)
        
        diff = abs(val1 - val2) if pd.notna(val1) and pd.notna(val2) else np.nan
        
        # Nur anzeigen wenn Differenz signifikant (> 0.001)
        if pd.notna(diff) and diff > 0.001:
             print(f"{row['config_set_name']:<30} | {val1:<10.4f} | {val2:<10.4f} | {diff:<10.4f}")

if __name__ == "__main__":
    # Dateinamen anpassen!
    f1 = "FINAL_AGGREGATED_RESULTS_UNSTR_357bit.csv"
    f2 = "FINAL_MERGED_RESULTS_UNSTR_357.csv"
    compare_csvs(f1, f2)