
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    path = "averaged_runs_summary_sorted_20250526Master.csv"
    return pd.read_csv(path)

def plot():
    df = load_data()
    df["delta_f1"] = df["fhe_simulate_json_f1_weighted_mean"] - df["pytorch_json_f1_weighted_mean"]
    df["delta_time"] = df["fhe_simulate_json_time_per_1000"] - df["pytorch_json_time_per_1000"]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="delta_time", y="delta_f1", hue="dataset_name", style="quant_bits")
    plt.axhline(0, color="gray", ls="--")
    plt.axvline(0, color="gray", ls="--")
    plt.title("Delta F1 vs. Delta Inference Time (FHE - PyTorch)")
    plt.xlabel("Δ Inference Time (FHE - PyTorch)")
    plt.ylabel("Δ F1 Score (FHE - PyTorch)")
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("delta_f1_vs_delta_time.png", dpi=150)
    plt.close()

if __name__ == '__main__':
    plot()