import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    path = "all_runs_detailed_data_20250526Master.csv"
    return pd.read_csv(path)

def plot():
    df = load_data()
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="dataset_name", y="fhe_simulate_json_f1_weighted")
    plt.title("FHE F1-Score Distribution by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("F1-Score (Weighted)")
    plt.grid(True, axis="y", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("fhe_f1_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="dataset_name", y="fhe_simulate_json_roc_auc_macro")
    plt.title("FHE ROC-AUC Distribution by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("ROC-AUC (Macro)")
    plt.grid(True, axis="y", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("fhe_roc_auc_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="dataset_name", y="fhe_simulate_json_time_per_1000", inner="box")
    plt.title("FHE Inference Time Distribution by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Inference Time per 1000 Samples (s)")
    plt.grid(True, axis="y", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("fhe_time_violinplot.png", dpi=150)
    plt.close()

if __name__ == '__main__':
    plot()