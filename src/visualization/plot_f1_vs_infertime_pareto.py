import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    path = "averaged_runs_summary_sorted_20250526Master.csv"
    return pd.read_csv(path)

def plot():
    df = load_data()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="unpruned_neurons", y="fhe_simulate_json_accuracy_mean",
                 hue="quant_bits", marker="o")
    plt.title("Impact of Pruning on FHE Accuracy")
    plt.xlabel("Unpruned Neurons")
    plt.ylabel("FHE Accuracy")
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("fhe_accuracy_vs_pruning.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="avg_final_sparsity_mean", y="fhe_simulate_json_time_per_1000",
                    hue="quant_bits", style="dataset_name")
    plt.title("Inference Time vs. Sparsity")
    plt.xlabel("Final Sparsity (mean)")
    plt.ylabel("FHE Inference Time per 1000 samples (s)")
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("fhe_time_vs_sparsity.png", dpi=150)
    plt.close()

if __name__ == '__main__':
    plot()