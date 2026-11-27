import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

LAND_PATH = "data/processed/khanhhoa_namtrungbo_land.gpkg"
CLUSTERS_PATH = "data/processed/khanhhoa_baseline_clusters.csv"
FIG_OUT = "docs/figures/fig3_baseline_comparison_khanhhoa.png"


def plot_comparison():
    land = gpd.read_file(LAND_PATH)
    df = pd.read_csv(CLUSTERS_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=300, sharex=True, sharey=True)
    configs = [
        ("cluster_std_dbscan", "A. DBSCAN chuẩn (Khoảng cách chim bay Haversine)"),
        ("cluster_ns_dbscan", "B. NS-DBSCAN tĩnh (Khoảng cách đường biển 8 hướng)"),
    ]

    for ax, (col, title) in zip(axes, configs):
        land.plot(ax=ax, color="#d9d2c5", edgecolor="#5c5346", linewidth=0.6, zorder=2)

        noise = df[df[col] == -1]
        clustered = df[df[col] != -1]
        n_clusters = clustered[col].nunique()

        ax.scatter(
            noise["node_lon"] + 0.005,
            noise["node_lat"] + 0.005,
            s=4,
            c="#bbbbbb",
            alpha=0.5,
            label=f"Nhiễu ({len(noise):,} nút)",
            zorder=1,
        )
        ax.scatter(
            clustered["node_lon"] + 0.005,
            clustered["node_lat"] + 0.005,
            s=10,
            c=clustered[col],
            cmap="tab20",
            alpha=0.85,
            label=f"Đã phân cụm ({n_clusters} cụm, {len(clustered):,} nút)",
            zorder=3,
        )

        ax.set_xlim(109.05, 109.85)
        ax.set_ylim(11.45, 13.15)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Kinh độ (°E)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="lower right", framealpha=0.95, fontsize=9)

    axes[0].set_ylabel("Vĩ độ (°N)")
    plt.suptitle(
        "So sánh phân cụm ngư trường lõi Khánh Hòa (GFW 2024, ≥20h/năm, ε = 3.5 km, MinPts = 18)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT)
    plt.close()
    print(f"Đã xuất biểu đồ đối chứng: {FIG_OUT}")


if __name__ == "__main__":
    plot_comparison()
