import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def prepare_data(file_path: str) -> pd.DataFrame:
    """Lädt und verarbeitet die CSV-Daten für die Analyse."""
    try:
        df_raw = pd.read_csv(file_path)
        df_raw.columns = df_raw.columns.str.replace('cfg_', '')
        df = df_raw[df_raw['dataset_name'].isin(['EdgeIIoT', 'CICIoT'])].copy()
        
        df_agg = df.groupby(['dataset_name', 'quant_bits', 'unpruned_neurons']).agg(
            f1_mean=('fhe_simulate_json_f1_weighted', 'mean'),
            fhe_time_mean=('fhe_simulate_json_time_per_1000', 'mean')
        ).reset_index()
        return df_agg
    
    except (FileNotFoundError, KeyError) as e:
        print(f"Fehler bei der Datenvorbereitung: {e}")
        return pd.DataFrame()

def plot_separate_pareto_fronts(df_aggregated: pd.DataFrame):
    """Berechnet für jeden Datensatz eine EIGENE Pareto-Front und stellt beide in einem gemeinsamen Plot dar, inklusive korrekter Legende."""
    
    df_edge = df_aggregated[df_aggregated['dataset_name'] == 'EdgeIIoT'].copy()
    df_ciciot = df_aggregated[df_aggregated['dataset_name'] == 'CICIoT'].copy()
    
    def find_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
        points = df[['fhe_time_mean', 'f1_mean']].values
        is_pareto = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            if np.any((points[:, 0] <= c[0]) & (points[:, 1] >= c[1]) & ((points[:, 0] < c[0]) | (points[:, 1] > c[1]))):
                is_pareto[i] = False
        return df[is_pareto].sort_values('fhe_time_mean')
        
    pareto_edge = find_pareto_front(df_edge)
    pareto_ciciot = find_pareto_front(df_ciciot)
    
    plt.figure(figsize=(10, 4))
    
    ax = sns.scatterplot(
        data=df_aggregated, x='fhe_time_mean', y='f1_mean',
        hue='dataset_name', style='dataset_name', s=100, alpha=0.7
    )
    ax.plot(pareto_edge['fhe_time_mean'], pareto_edge['f1_mean'],
            color='red', linestyle='--', marker='D', markersize=4, zorder=5, label='Pareto-Front (EdgeIIoT)')
    ax.plot(pareto_ciciot['fhe_time_mean'], pareto_ciciot['f1_mean'],
            color='red', linestyle=':', marker='D', markersize=4, zorder=5, label='Pareto-Front (CICIoT)')
    
    for _, row in df_aggregated.iterrows():
        plt.annotate(f"({row['quant_bits']}b, {int(row['unpruned_neurons'])}n)",
                     (row['fhe_time_mean'], row['f1_mean']),
                     xytext=(6, 0), textcoords='offset points',
                     fontsize=7, alpha=0.8)
    
    plt.title('F1-Score vs. FHE Inference Time with Pareto Fronts', fontsize=12)
    plt.xlabel('FHE Inference Time (s / 1000 samples)', fontsize=10)
    plt.ylabel('F1-Score (Weighted)', fontsize=10)
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(title='Legende', fontsize=8)
    #plt.savefig('pareto_fronts.png', dpi=300, bbox_inches='tight')
    plt.savefig('pareto_fronts.pdf', dpi=300, bbox_inches='tight', format="pdf")
    plt.close()
    print("\nPlot mit korrigierter Legende 'pareto_fronts.png' gespeichert.")

if __name__ == '__main__':
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        data_file_path = os.path.join(
            project_root, 'results', 'qat_runs', 'all_runs_detailed_data_20250526101416Master.csv'
        )
    except NameError:
        data_file_path = os.path.join(
            'results', 'qat_runs', 'all_runs_detailed_data_20250526101416Master.csv'
        )
    aggregated_data = prepare_data(data_file_path)
    if not aggregated_data.empty:
        plot_separate_pareto_fronts(aggregated_data)
    print("\nAnalyse-Skript beendet.")
