
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    path = "averaged_runs_summary_sorted_20250526Master.csv"
    return pd.read_csv(path)

def plot_compile_time():
    df = load_data()
    df_edge = df[df["dataset_name"] == "CICIoT"]

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_edge,
                 x="unpruned_neurons",
                 y="fhe_compilation_time_s_mean",
                 hue="quant_bits",
                 marker="o")

    plt.title("Compile Time vs. Unpruned Neurons (CICIoT)")
    plt.xlabel("Unpruned Neurons")
    plt.ylabel("FHE Compile Time (s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("Compile Time vs. Unpruned Neurons (CICIoT).png", dpi=150)
    plt.close()

if __name__ == '__main__':
    plot_compile_time()
