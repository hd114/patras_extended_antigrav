# pareto_plot.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob # Importiere glob für die Dateisuche
import sys

def prepare_data(file_path: str) -> pd.DataFrame:
    """
    Lädt und verarbeitet die CSV-Daten für die Analyse.
    Verwendet jetzt explizit die Spalten aus der Re-Evaluierung.
    """
    try:
        df_raw = pd.read_csv(file_path, sep=',')
        
        # HINWEIS: Das Entfernen von 'cfg_' ist nicht mehr nötig, 
        # da der neue Analyzer dies bereits bei der Aggregation tut.
        # Für die "detailed" CSV behalten wir es bei, um Konsistenz zu gewährleisten.
        if any(col.startswith('cfg_') for col in df_raw.columns):
            df_raw.columns = df_raw.columns.str.replace('cfg_', '')
        
        df = df_raw[df_raw['dataset_name'].isin(['EdgeIIoT', 'CICIoT'])].copy()
        
        # --- ANPASSUNG ---
        # Dieser Block stellt sicher, dass die korrekten Spalten für den Plot verwendet werden.
        # Wir greifen jetzt auf die Spalten mit dem Präfix 'fhe_reeval_execute_txt' zu.
        f1_col = 'fhe_reeval_execute_txt_F1w'
        time_col = 'fhe_reeval_execute_txt_time_per_1000_samples_s_txt'

        # Überprüfen, ob die Spalten vorhanden sind
        if f1_col not in df.columns or time_col not in df.columns:
            print(f"FEHLER: Notwendige Spalten nicht gefunden: '{f1_col}' oder '{time_col}'.")
            print("Verfügbare Spalten:", df.columns.tolist())
            return pd.DataFrame()

        # Erstelle die für den Plot benötigten Spalten
        df['f1_for_plot'] = df[f1_col]
        df['time_for_plot'] = df[time_col]
        
        # Aggregiere die Daten über Wiederholungen, um Mittelwerte zu erhalten
        # Gruppierungsvariablen müssen im DataFrame vorhanden sein
        grouping_vars = ['dataset_name', 'quant_bits', 'unpruned_neurons']
        df_agg = df.groupby(grouping_vars).agg(
            f1_mean=('f1_for_plot', 'mean'),
            fhe_time_mean=('time_for_plot', 'mean')
        ).reset_index()
        return df_agg
    
    except FileNotFoundError:
        print(f"Fehler: Die Datei '{file_path}' wurde nicht gefunden.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
        return pd.DataFrame()

def plot_separate_pareto_fronts(df_aggregated: pd.DataFrame):
    """
    Berechnet für jeden Datensatz eine EIGENE Pareto-Front und stellt beide 
    in einem gemeinsamen Plot dar.
    """
    if df_aggregated.empty:
        print("Keine Daten zum Plotten vorhanden.")
        return

    df_edge = df_aggregated[df_aggregated['dataset_name'] == 'EdgeIIoT'].copy()
    df_ciciot = df_aggregated[df_aggregated['dataset_name'] == 'CICIoT'].copy()
    
    def find_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
        """Findet die Pareto-optimalen Punkte in einem DataFrame."""
        points = df[['fhe_time_mean', 'f1_mean']].values
        if len(points) == 0:
            return pd.DataFrame(columns=df.columns)
        
        is_pareto = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            # Ein Punkt ist nicht Pareto-optimal, wenn ein anderer Punkt
            # in beiden Dimensionen besser oder gleich ist (und in einer Dimension strikt besser).
            # Besser = niedrigere Zeit (points[:, 0]) UND höherer F1 (points[:, 1])
            if np.any((points[:, 0] <= c[0]) & (points[:, 1] >= c[1]) & ((points[:, 0] < c[0]) | (points[:, 1] > c[1]))):
                is_pareto[i] = False
        return df[is_pareto].sort_values('fhe_time_mean')
    
    pareto_edge = find_pareto_front(df_edge)
    pareto_ciciot = find_pareto_front(df_ciciot)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 5))
    
    ax = sns.scatterplot(
        data=df_aggregated, x='fhe_time_mean', y='f1_mean',
        hue='dataset_name', style='dataset_name', s=150, alpha=0.7, zorder=10
    )
    
    # Plotten der Pareto-Fronten
    if not pareto_edge.empty:
        ax.plot(pareto_edge['fhe_time_mean'], pareto_edge['f1_mean'],
                color='red', linestyle='--', marker='D', markersize=5, zorder=5, label='Pareto-Front (EdgeIIoT)')
    if not pareto_ciciot.empty:
        ax.plot(pareto_ciciot['fhe_time_mean'], pareto_ciciot['f1_mean'],
                color='blue', linestyle=':', marker='X', markersize=6, zorder=5, label='Pareto-Front (CICIoT)')
    
    # Annotieren der Punkte mit Konfigurationsdetails
    for _, row in df_aggregated.iterrows():
        plt.annotate(f"({row['quant_bits']}b, {int(row['unpruned_neurons'])}n)",
                     (row['fhe_time_mean'], row['f1_mean']),
                     xytext=(6, 0), textcoords='offset points',
                     fontsize=8, alpha=0.9,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.5))
    
    plt.title('F1-Score vs. FHE Inference Time with Pareto Fronts', fontsize=14)
    plt.xlabel('FHE Inference Time (s / 1000 samples)', fontsize=12)
    plt.ylabel('F1-Score (Weighted)', fontsize=12)
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    
    # Legende anpassen
    handles, labels = ax.get_legend_handles_labels()
    # Entferne Duplikate aus der Legende (z.B. wenn Pareto-Fronten eigene Labels haben)
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Legende', fontsize=10)
    
    plt.tight_layout()
    
    # Speichern in verschiedenen Formaten
    output_filename_base = 'pareto_fronts_reeval_execute_mode'
    plt.savefig(f'{output_filename_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_filename_base}.pdf', bbox_inches='tight')
    plt.close()
    print(f"\nPlot '{output_filename_base}.png' und '.pdf' wurden gespeichert.")

if __name__ == '__main__':
    """
    Haupteinstiegspunkt des Skripts.
    Verarbeitet Kommandozeilen-Argumente für den Dateipfad, sucht andernfalls
    automatisch nach der neuesten CSV-Datei oder fordert zur Eingabe auf.
    """
    data_file_path = None

    # --- ANFANG DER ANPASSUNG ---
    # Block zur Verarbeitung von Kommandozeilen-Argumenten.
    # Prüft, ob nach dem Skriptnamen (sys.argv[0]) ein weiteres Argument übergeben wurde.
    if len(sys.argv) > 1:
        # Das erste Argument nach dem Skriptnamen wird als Dateipfad verwendet.
        data_file_path = sys.argv[1]
        print(f"INFO: Verwende CSV-Datei aus Kommandozeilen-Argument: {data_file_path}")
    else:
        # Wenn kein Argument übergeben wurde, wird die bisherige Logik ausgeführt.
        print("INFO: Kein Kommandozeilen-Argument angegeben. Suche automatisch nach CSV-Datei...")
        default_data_file = None
        current_dir = os.getcwd()
        possible_paths = [
            current_dir,
            os.path.join(current_dir, 'results'),
            os.path.join(current_dir, '..', 'results', 'qat_runs'),
            os.path.join(current_dir, 'results', 'qat_runs')
        ]
        
        csv_files = []
        for path in possible_paths:
            if os.path.isdir(path):
                files = glob.glob(os.path.join(path, "all_runs_detailed_data_*.csv"))
                csv_files.extend(files)

        if csv_files:
            default_data_file = max(csv_files, key=os.path.getctime)
            print(f"INFO: Neueste gefundene Datendatei wird verwendet: {default_data_file}")
            data_file_path = default_data_file
        else:
            print("FEHLER: Konnte keine 'all_runs_detailed_data_*.csv' Datei finden.")
            data_file_path = input("Bitte den vollständigen Pfad zur CSV-Datei eingeben: ").strip()
    # --- ENDE DER ANPASSUNG ---

    # Führt die Analyse mit dem ermittelten Dateipfad aus.
    if data_file_path and os.path.isfile(data_file_path):
        aggregated_data = prepare_data(data_file_path)
        if not aggregated_data.empty:
            plot_separate_pareto_fronts(aggregated_data)
        else:
            print("Keine verarbeitbaren Daten für den Plot gefunden.")
    else:
        print(f"FEHLER: Datei nicht gefunden oder kein Pfad angegeben: '{data_file_path}'")
        
    print("\nPareto-Plot-Skript beendet.")