import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm

def load_data():
    path = "averaged_runs_summary_sorted_20250526Master.csv"
    return pd.read_csv(path)

def plot_heatmap(df, metric, dataset, output_file):
    pivot = df.pivot_table(index="unpruned_neurons", columns="quant_bits", values=metric)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"Heatmap of {metric} ({dataset})")
    plt.xlabel("Quantization Bits")
    plt.ylabel("Unpruned Neurons")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_3d_surface(df, metric, dataset, output_file):
    df_sorted = df.sort_values(["quant_bits", "unpruned_neurons"])
    bits = sorted(df["quant_bits"].unique())
    neurons = sorted(df["unpruned_neurons"].unique())
    X, Y = np.meshgrid(bits, neurons)
    Z = df.pivot_table(index="unpruned_neurons", columns="quant_bits", values=metric).values

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor="k")
    ax.set_xlabel("Quantization Bits")
    ax.set_ylabel("Unpruned Neurons")
    ax.set_zlabel(metric.replace("_", " ").capitalize())
    ax.set_title(f"3D Surface of {metric} ({dataset})")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot():
    df = load_data()
    metrics = ["fhe_simulate_json_f1_weighted_mean", "fhe_simulate_json_roc_auc_macro_mean"]
    for dataset in ["EdgeIIoT", "CICIoT"]:
        df_subset = df[df["dataset_name"] == dataset]
        for metric in metrics:
            suffix = metric.split("_")[0]
            heatmap_filename = f"Heatmap of {metric} ({dataset}).png"
            surface_filename = f"3D Surface of {metric} ({dataset}).png"
            plot_heatmap(df_subset, metric, dataset, heatmap_filename)
            plot_3d_surface(df_subset, metric, dataset, surface_filename)

if __name__ == '__main__':
    plot()
