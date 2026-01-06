# src/visualization/plots.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns # Für verbesserte Ästhetik und Paletten
import os
from typing import List, Optional, Tuple

# PEP8 Konforme Konstanten
OUTPUT_PLOT_DIR_NAME = "concreteML_plots" # Name für das Output-Verzeichnis
PLOT_DPI = 300 
USE_SEABORN_STYLE = True

# Konsistente Farbpalette (Colorblind-friendly von Seaborn)
# Diese Palette wird für die verschiedenen Plots verwendet.
COLOR_PALETTE = sns.color_palette("colorblind", n_colors=10)
# Beispiel-Marker für Scatter-Plots, falls benötigt
MARKER_STYLES = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']


def load_data(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Lädt die Daten aus der angegebenen CSV-Datei und konsolidiert Dataset-Namen.

    Args:
        csv_path (str): Pfad zur CSV-Datei.

    Returns:
        Optional[pd.DataFrame]: Geladenes DataFrame oder None bei Fehlern.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Daten erfolgreich von '{csv_path}' geladen.")
        
        # Datensatz-Namen konsolidieren
        if 'dataset_name' in df.columns:
            df['dataset_name'] = df['dataset_name'].replace(['CiIoT', 'CICIoT'], 'CICIoT23')
            print(f"Dataset-Namen konsolidiert. Eindeutige Werte jetzt: {df['dataset_name'].unique()}")
        else:
            print("WARNUNG: Spalte 'dataset_name' nicht im DataFrame gefunden für Konsolidierung.")

        print(f"Spalten im DataFrame: {df.columns.tolist()}")
        print(f"Anzahl der Zeilen: {len(df)}")
        
        required_cols = ['dataset_name', 'quantization_bits', 'unpruned_neurons', 
                         'best_f1_weighted_val', 'pytorch_eval_f1_weighted', 
                         'fhe_simulate_f1_weighted', 'pytorch_eval_time_per_1000',
                         'fhe_simulate_time_per_1000']
        for col in required_cols:
            if col not in df.columns:
                print(f"WARNUNG: Erforderliche Spalte '{col}' nicht im DataFrame gefunden.")
        return df
    except FileNotFoundError:
        print(f"FEHLER: CSV-Datei nicht gefunden unter: {csv_path}")
        return None
    except Exception as e:
        print(f"FEHLER beim Laden der CSV-Datei: {e}")
        return None


def setup_ieee_plot_style():
    """Konfiguriert einen Plot-Stil, der für IEEE-Papers geeignet ist."""
    if USE_SEABORN_STYLE:
        sns.set_theme(context='paper', style="whitegrid", font_scale=1.1) # Etwas kleinere font_scale für kompaktere Plots
    else: 
        plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif'] # Times New Roman oft bevorzugt
    plt.rcParams['axes.labelsize'] = 9 # Achsenbeschriftungen
    plt.rcParams['axes.titlesize'] = 10 # Titel von Subplots
    plt.rcParams['xtick.labelsize'] = 8 
    plt.rcParams['ytick.labelsize'] = 8 
    plt.rcParams['legend.fontsize'] = 8 
    plt.rcParams['legend.title_fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 11 # Haupttitel der Figure
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 5
    plt.rcParams['figure.dpi'] = PLOT_DPI
    plt.rcParams['savefig.dpi'] = PLOT_DPI
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['axes.grid'] = True


def plot_f1_vs_quantization(df: pd.DataFrame, output_dir: str):
    """
    Erstellt einen Plot: F1-Score (PyTorch, FHE) vs. Quantization Bits, gruppiert nach Datensatz.
    """
    if not all(col in df.columns for col in ['dataset_name', 'quantization_bits', 
                                             'pytorch_eval_f1_weighted', 'fhe_simulate_f1_weighted']):
        print("WARNUNG (plot_f1_vs_quantization): Notwendige Spalten fehlen, Plot wird übersprungen.")
        return

    df_plot = df.dropna(subset=['quantization_bits', 'pytorch_eval_f1_weighted', 'fhe_simulate_f1_weighted'])
    if df_plot.empty:
        print("WARNUNG (plot_f1_vs_quantization): Keine Daten nach dem Entfernen von NaNs für den Plot.")
        return

    df_melted = df_plot.melt(
        id_vars=['dataset_name', 'quantization_bits', 'unpruned_neurons'], # unpruned_neurons behalten für ggf. tiefere Aggregation
        value_vars=['pytorch_eval_f1_weighted', 'fhe_simulate_f1_weighted'],
        var_name='Evaluation Type',
        value_name='F1 Score (Weighted)'
    )
    df_melted['Evaluation Type'] = df_melted['Evaluation Type'].replace({
        'pytorch_eval_f1_weighted': 'PyTorch Eval',
        'fhe_simulate_f1_weighted': 'FHE Simulate'
    })

    df_melted['quantization_bits'] = df_melted['quantization_bits'].astype('category')
    
    # Aggregiere über 'unpruned_neurons', um einen Wert pro (dataset, qbits, eval_type) zu bekommen
    df_plot_agg = df_melted.groupby(
        ['dataset_name', 'quantization_bits', 'Evaluation Type'],
        as_index=False, observed=True 
    )['F1 Score (Weighted)'].mean()

    palette = {"PyTorch Eval": COLOR_PALETTE[0], "FHE Simulate": COLOR_PALETTE[1]}

    g = sns.catplot(
        data=df_plot_agg,
        x='quantization_bits',
        y='F1 Score (Weighted)',
        hue='Evaluation Type',
        col='dataset_name',
        kind='bar',
        palette=palette,
        height=3.5, aspect=1.1, # Kompaktere Plots
        legend=False # Legende wird manuell hinzugefügt
    )
    
    g.set_axis_labels("Quantization Bits", "F1 Score (Weighted)")
    g.set_titles("Dataset: {col_name}")
    g.despine(left=True)
    
    # Hauptlegende für die gesamte Figur hinzufügen
    handles, labels = plt.gca().get_legend_handles_labels() # Hole Handles vom letzten Subplot
    g.fig.legend(handles, labels, title='Evaluation Type', loc='upper center', 
                 bbox_to_anchor=(0.5, 0.05), ncol=2, frameon=True) # Unter dem Plot zentriert
                 
    g.fig.suptitle("F1 Score vs. Quantization Bits by Dataset and Evaluation Type", y=1.02, fontsize=plt.rcParams['figure.titlesize'])
    g.fig.tight_layout(rect=[0, 0.05, 1, 1]) # Platz für Legende unten lassen

    plot_filename = os.path.join(output_dir, "f1_vs_quantization_bits.png")
    plt.savefig(plot_filename, dpi=PLOT_DPI)
    plt.close()
    print(f"Plot gespeichert: {plot_filename}")


def plot_fhe_time_vs_quantization(df: pd.DataFrame, output_dir: str):
    """
    Erstellt einen Plot: FHE Inferenzzeit vs. Quantization Bits, gruppiert nach Datensatz.
    """
    if not all(col in df.columns for col in ['dataset_name', 'quantization_bits', 'fhe_simulate_time_per_1000']):
        print("WARNUNG (plot_fhe_time_vs_quantization): Notwendige Spalten fehlen, Plot wird übersprungen.")
        return
        
    df_plot = df.dropna(subset=['quantization_bits', 'fhe_simulate_time_per_1000'])
    if df_plot.empty:
        print("WARNUNG (plot_fhe_time_vs_quantization): Keine Daten nach dem Entfernen von NaNs für den Plot.")
        return

    df_plot['quantization_bits'] = df_plot['quantization_bits'].astype('category')
    
    df_plot_agg = df_plot.groupby(
        ['dataset_name', 'quantization_bits'], 
        as_index=False, observed=True
    )['fhe_simulate_time_per_1000'].mean()
    
    g = sns.catplot(
        data=df_plot_agg,
        x='quantization_bits',
        y='fhe_simulate_time_per_1000',
        col='dataset_name',
        kind='bar',
        palette=COLOR_PALETTE[2:5], # Verwende eine konsistente Teilpalette
        height=3.5, aspect=1.1
    )
    
    g.set_axis_labels("Quantization Bits", "FHE Inf. Time / 1000 Samples (s)") # Klarere Achsenbeschriftung
    g.set_titles("Dataset: {col_name}")
    g.despine(left=True)
    g.fig.suptitle("FHE Inference Time vs. Quantization Bits by Dataset", y=1.02, fontsize=plt.rcParams['figure.titlesize'])
    g.fig.tight_layout()

    plot_filename = os.path.join(output_dir, "fhe_time_vs_quantization_bits.png")
    plt.savefig(plot_filename, dpi=PLOT_DPI)
    plt.close()
    print(f"Plot gespeichert: {plot_filename}")


def plot_f1_vs_unpruned_neurons(df: pd.DataFrame, output_dir: str, fixed_qbits: Optional[int] = None):
    """
    Erstellt einen Plot: F1-Score (FHE) vs. Unpruned Neurons,
    optional für feste Quantization Bits, gruppiert nach Datensatz.
    """
    if not all(col in df.columns for col in ['dataset_name', 'quantization_bits', 'unpruned_neurons', 
                                             'fhe_simulate_f1_weighted']):
        print("WARNUNG (plot_f1_vs_unpruned_neurons): Notwendige Spalten fehlen, Plot wird übersprungen.")
        return

    df_plot = df.copy()
    title_suffix = ""
    filename_suffix = ""

    if fixed_qbits is not None:
        if 'quantization_bits' in df_plot.columns:
            df_plot = df_plot[df_plot['quantization_bits'] == fixed_qbits]
            title_suffix = f" (Quant. Bits: {fixed_qbits})" # Präziserer Titel
            filename_suffix = f"_qb{fixed_qbits}"
        else:
            print(f"WARNUNG (plot_f1_vs_unpruned_neurons): Spalte 'quantization_bits' für Filterung nicht vorhanden.")
            
    df_plot = df_plot.dropna(subset=['unpruned_neurons', 'fhe_simulate_f1_weighted'])
    if df_plot.empty:
        print(f"WARNUNG (plot_f1_vs_unpruned_neurons): Keine Daten für Plot nach Filterung (qbits={fixed_qbits}).")
        return

    # Stelle sicher, dass unpruned_neurons als Kategorie für korrekte Sortierung/Darstellung behandelt wird
    # und in der gewünschten Reihenfolge (4, 8, 16, 32) sortiert wird, falls es als Zahl geladen wurde.
    if pd.api.types.is_numeric_dtype(df_plot['unpruned_neurons']):
        df_plot = df_plot.sort_values(by='unpruned_neurons')
    df_plot['unpruned_neurons'] = df_plot['unpruned_neurons'].astype('category')


    g = sns.catplot(
        data=df_plot,
        x='unpruned_neurons',
        y='fhe_simulate_f1_weighted',
        col='dataset_name',
        kind='bar',
        palette=COLOR_PALETTE[3:7], # Eigene Farbpalette für diesen Plot-Typ
        height=3.5, aspect=1.1,
        # errorbar=None # Falls Sie Fehlerbalken (basierend auf Wiederholungen) hätten
    )
    
    g.set_axis_labels("Unpruned Neurons per Layer", "FHE F1 Score (Weighted)")
    g.set_titles("Dataset: {col_name}") # Titel für jeden Subplot
    g.despine(left=True)
    # Haupttitel für die gesamte Figur
    g.fig.suptitle(f"FHE F1 Score vs. Unpruned Neurons{title_suffix}", y=1.02, fontsize=plt.rcParams['figure.titlesize'])
    g.fig.tight_layout()

    plot_filename = os.path.join(output_dir, f"fhe_f1_vs_unpruned_neurons{filename_suffix}.png")
    plt.savefig(plot_filename, dpi=PLOT_DPI)
    plt.close()
    print(f"Plot gespeichert: {plot_filename}")


def plot_pytorch_vs_fhe_f1_scatter(df: pd.DataFrame, output_dir: str):
    """
    Erstellt einen Scatter-Plot: PyTorch F1 vs. FHE F1,
    mit Punkten gefärbt nach Quantization Bits und Form nach Dataset.
    """
    if not all(col in df.columns for col in ['pytorch_eval_f1_weighted', 'fhe_simulate_f1_weighted', 
                                             'quantization_bits', 'dataset_name', 'unpruned_neurons']):
        print("WARNUNG (plot_pytorch_vs_fhe_f1_scatter): Notwendige Spalten fehlen, Plot wird übersprungen.")
        return

    df_plot = df.dropna(subset=['pytorch_eval_f1_weighted', 'fhe_simulate_f1_weighted', 
                                 'quantization_bits', 'dataset_name', 'unpruned_neurons'])
    if df_plot.empty:
        print("WARNUNG (plot_pytorch_vs_fhe_f1_scatter): Keine Daten für Scatter-Plot nach NaNs.")
        return

    plt.figure(figsize=(6, 5)) # Etwas kompakter
    df_plot['quantization_bits_cat'] = df_plot['quantization_bits'].astype('category')
    
    # Verwende eine spezifische Palette für Quantisierungsbits
    hue_palette = {
        qbit: COLOR_PALETTE[i % len(COLOR_PALETTE)] 
        for i, qbit in enumerate(sorted(df_plot['quantization_bits_cat'].unique()))
    }

    scatter_plot = sns.scatterplot(
        data=df_plot,
        x='pytorch_eval_f1_weighted',
        y='fhe_simulate_f1_weighted',
        hue='quantization_bits_cat', # Verwende kategorische Spalte für Hue
        style='dataset_name', 
        size='unpruned_neurons', 
        sizes=(40, 150), 
        palette=hue_palette, 
        legend="full" 
    )
    
    lim_min = df_plot[['pytorch_eval_f1_weighted', 'fhe_simulate_f1_weighted']].min().min()
    lim_max = df_plot[['pytorch_eval_f1_weighted', 'fhe_simulate_f1_weighted']].max().max()
    lims = [
        max(0, lim_min - 0.05),
        min(1, lim_max + 0.05),
    ]

    plt.plot(lims, lims, 'k--', alpha=0.6, zorder=0, label='Ideal (PyTorch = FHE)')
    
    plt.xlim(lims)
    plt.ylim(lims)
    
    plt.xlabel("PyTorch F1 Score (Weighted)")
    plt.ylabel("FHE Simulate F1 Score (Weighted)")
    plt.title("PyTorch vs. FHE Simulated F1 Score", fontsize=plt.rcParams['figure.titlesize']-1) # Titel etwas kleiner
    
    # Legende anpassen für bessere Lesbarkeit
    handles, labels = scatter_plot.get_legend_handles_labels()
    # Manchmal enthält die Legende Titel für hue, style, size. Wir wollen sie gruppieren.
    # Dies erfordert oft manuelle Anpassung oder ist komplex mit Seaborn's automatischer Legende.
    # Fürs Erste: Standardlegende außerhalb.
    plt.legend(title='Parameters', bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.82, 1]) # Mehr Platz für Legende

    plot_filename = os.path.join(output_dir, "pytorch_vs_fhe_f1_scatter.png")
    plt.savefig(plot_filename, dpi=PLOT_DPI)
    plt.close()
    print(f"Plot gespeichert: {plot_filename}")


def plot_fhe_tradeoff_f1_vs_time(df: pd.DataFrame, output_dir: str):
    """
    Erstellt einen Scatter-Plot für den Trade-off: FHE F1 Score vs. FHE Inference Time.
    """
    if not all(col in df.columns for col in ['fhe_simulate_f1_weighted', 'fhe_simulate_time_per_1000',
                                             'quantization_bits', 'dataset_name', 'unpruned_neurons']):
        print("WARNUNG (plot_fhe_tradeoff_f1_vs_time): Notwendige Spalten fehlen, Plot wird übersprungen.")
        return

    df_plot = df.dropna(subset=['fhe_simulate_f1_weighted', 'fhe_simulate_time_per_1000',
                                 'quantization_bits', 'dataset_name', 'unpruned_neurons'])
    if df_plot.empty:
        print("WARNUNG (plot_fhe_tradeoff_f1_vs_time): Keine Daten für Trade-off-Plot nach NaNs.")
        return

    plt.figure(figsize=(7, 5)) # Etwas kompakter
    df_plot['quantization_bits_cat'] = df_plot['quantization_bits'].astype('category')
    
    hue_palette = {
        qbit: COLOR_PALETTE[i % len(COLOR_PALETTE)] 
        for i, qbit in enumerate(sorted(df_plot['quantization_bits_cat'].unique()))
    }

    scatter_plot = sns.scatterplot(
        data=df_plot,
        x='fhe_simulate_time_per_1000',
        y='fhe_simulate_f1_weighted',
        hue='quantization_bits_cat',
        style='dataset_name',
        size='unpruned_neurons',
        sizes=(40, 150),
        palette=hue_palette,
        legend="full"
    )

    plt.xlabel("FHE Inf. Time / 1000 Samples (s) [log scale]") # Klarere Achsenbeschriftung
    plt.ylabel("FHE Simulate F1 Score (Weighted)")
    plt.title("FHE: F1 Score vs. Inference Time Trade-off", fontsize=plt.rcParams['figure.titlesize']-1)
    
    min_x_val = df_plot['fhe_simulate_time_per_1000'].min()
    if min_x_val is not None and pd.notna(min_x_val) and min_x_val > 0 : 
        plt.xscale('log')
        scatter_plot.xaxis.set_major_formatter(mticker.ScalarFormatter())
        # scatter_plot.xaxis.get_major_formatter().set_scientific(False) # Optional
        # scatter_plot.xaxis.get_major_formatter().set_useOffset(False) # Optional
    else:
        print("WARNUNG (plot_fhe_tradeoff_f1_vs_time): Nicht-positive oder fehlende Werte für Inferenzzeit, Log-Skala möglicherweise nicht optimal.")

    plt.legend(title='Parameters', bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    plot_filename = os.path.join(output_dir, "fhe_tradeoff_f1_vs_time.png")
    plt.savefig(plot_filename, dpi=PLOT_DPI)
    plt.close()
    print(f"Plot gespeichert: {plot_filename}")


def main():
    """
    Hauptfunktion zum Laden der Daten und Erstellen der Plots.
    """
    print("Starte Plot-Analyse für IEEE Paper...")
    setup_ieee_plot_style()

    script_path = os.path.abspath(__file__)
    src_visualization_dir = os.path.dirname(script_path)
    src_dir = os.path.dirname(src_visualization_dir)
    project_root = os.path.dirname(src_dir)

    csv_file_name = "averaged_runs_summary_sorted.csv"
    csv_file_path = os.path.join(project_root, "results", "qat_runs", csv_file_name)
    
    output_plot_full_path = os.path.join(project_root, OUTPUT_PLOT_DIR_NAME)

    if not os.path.exists(output_plot_full_path):
        os.makedirs(output_plot_full_path)
        print(f"Output-Verzeichnis '{output_plot_full_path}' erstellt.")

    data_df = load_data(csv_file_path)

    if data_df is None or data_df.empty:
        print("Keine Daten zum Plotten vorhanden. Skript wird beendet.")
        return

    print("\nErstelle Plots...")
    plot_f1_vs_quantization(data_df.copy(), output_plot_full_path)
    plot_fhe_time_vs_quantization(data_df.copy(), output_plot_full_path)
    
    if 'quantization_bits' in data_df.columns:
        unique_qbits = sorted(data_df['quantization_bits'].dropna().unique())
        unique_qbits_int = [int(q) for q in unique_qbits if pd.notna(q) and q.is_integer()] # Stellt sicher, dass es ganze Zahlen sind
        for qbit_val in unique_qbits_int:
            plot_f1_vs_unpruned_neurons(data_df.copy(), output_plot_full_path, fixed_qbits=qbit_val)
    else:
         plot_f1_vs_unpruned_neurons(data_df.copy(), output_plot_full_path) 

    plot_pytorch_vs_fhe_f1_scatter(data_df.copy(), output_plot_full_path)
    plot_fhe_tradeoff_f1_vs_time(data_df.copy(), output_plot_full_path)

    print(f"\nAlle Plots wurden erstellt und im Verzeichnis '{output_plot_full_path}' gespeichert.")
    print("Skript beendet.")


if __name__ == "__main__":
    main()