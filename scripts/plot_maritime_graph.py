import os
import pickle

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

GRAPH_PATH = "data/processed/maritime_graph_8dir.pkl"
LAND_PATH = "data/processed/khanhhoa_namtrungbo_land.gpkg"
GFW_CSV_PATH = "data/processed/KhanhHoa_NamTrungBo_2024.csv"
FIG_DIR = "docs/figures"


def load_resources():
    with open(GRAPH_PATH, "rb") as f:
        data = pickle.load(f)
    land = gpd.read_file(LAND_PATH)
    return data["graph"], data["sea_mask"], data["bounds"], land


def build_sea_kdtree(G):
    nodes = list(G.nodes())
    coords_rad = np.radians([(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in nodes])
    return nodes, cKDTree(coords_rad)


def snap_to_sea_node(lat, lon, nodes, tree):
    _, idx = tree.query(np.radians([lat, lon]))
    return nodes[int(idx)]


def plot_study_area_and_snapping(G, sea_mask, bounds, land):
    """Hình 1: Toàn cảnh vùng biển Khánh Hòa - Nam Trung Bộ & 138 ô sát mép bờ."""
    lat_min, lat_max, lon_min, lon_max, step = bounds
    df = pd.read_csv(
        GFW_CSV_PATH, usecols=["cell_ll_lat", "cell_ll_lon", "fishing_hours"]
    )
    cells = df.groupby(["cell_ll_lat", "cell_ll_lon"], as_index=False)[
        "fishing_hours"
    ].sum()

    r_idx = np.round((cells["cell_ll_lat"] - lat_min) / step).astype(int)
    c_idx = np.round((cells["cell_ll_lon"] - lon_min) / step).astype(int)
    in_sea = sea_mask[r_idx, c_idx]

    sea_cells = cells[in_sea]
    shore_cells = cells[~in_sea]

    fig, ax = plt.subplots(figsize=(9, 10), dpi=300)
    land.plot(ax=ax, color="#d9d2c5", edgecolor="#5c5346", linewidth=0.6, zorder=2)

    ax.scatter(
        sea_cells["cell_ll_lon"] + 0.005,
        sea_cells["cell_ll_lat"] + 0.005,
        s=1.2,
        c="#1f77b4",
        alpha=0.45,
        label=f"Ô AIS trên biển ({len(sea_cells):,} ô - 99.69%)",
        zorder=1,
    )
    ax.scatter(
        shore_cells["cell_ll_lon"] + 0.005,
        shore_cells["cell_ll_lat"] + 0.005,
        s=18,
        c="#d62728",
        marker="x",
        linewidths=0.9,
        label=f"Ô sát mép bờ cần ánh xạ ({len(shore_cells):,} ô - 0.31%)",
        zorder=3,
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title(
        "Phân bố 45.082 ô lưới AIS GFW 2024 & Đường bờ biển OSM (Khánh Hòa - Nam Trung Bộ)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Kinh độ (°E)")
    ax.set_ylabel("Vĩ độ (°N)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)

    out_path = os.path.join(FIG_DIR, "fig1_gfw_osm_alignment.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Đã xuất: {out_path}")


def plot_coastal_detour(G, land):
    """Hình 2: So sánh đường chim bay vs đường biển né bán đảo Hòn Hèo & Cam Ranh."""
    nodes, tree = build_sea_kdtree(G)
    routes = [
        (
            "Vịnh Vân Phong ↔ Đầm Nha Phu\n(Chim bay: 30.19 km | Đường biển: 56.64 km, +87.6%)",
            (12.64, 109.32),
            (12.40, 109.19),
            "#e6550d",
        ),
        (
            "Vịnh Nha Trang ↔ Vịnh Cam Ranh\n(Chim bay: 36.95 km | Đường biển: 53.44 km, +44.6%)",
            (12.21, 109.23),
            (11.88, 109.20),
            "#2ca02c",
        ),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 9.5), dpi=300)
    land.plot(ax=ax, color="#d9d2c5", edgecolor="#5c5346", linewidth=0.7, zorder=2)

    for label, (lat_a, lon_a), (lat_b, lon_b), color in routes:
        node_a = snap_to_sea_node(lat_a, lon_a, nodes, tree)
        node_b = snap_to_sea_node(lat_b, lon_b, nodes, tree)
        path_nodes = nx.shortest_path(G, source=node_a, target=node_b, weight="weight")

        path_lons = [G.nodes[n]["lon"] + 0.005 for n in path_nodes]
        path_lats = [G.nodes[n]["lat"] + 0.005 for n in path_nodes]

        # Đường chim bay (nét đứt)
        ax.plot(
            [path_lons[0], path_lons[-1]],
            [path_lats[0], path_lats[-1]],
            linestyle="--",
            color="#333333",
            linewidth=1.5,
            zorder=4,
        )
        # Đường biển thực tế trên đồ thị 8 hướng (nét liền)
        ax.plot(
            path_lons,
            path_lats,
            linestyle="-",
            color=color,
            linewidth=2.4,
            label=label,
            zorder=5,
        )
        ax.scatter(
            [path_lons[0], path_lons[-1]],
            [path_lats[0], path_lats[-1]],
            color=color,
            s=45,
            edgecolor="black",
            zorder=6,
        )

    ax.plot(
        [],
        [],
        linestyle="--",
        color="#333333",
        linewidth=1.5,
        label="Đường chim bay Haversine (cắt xuyên đất liền)",
    )
    ax.set_xlim(109.05, 109.55)
    ax.set_ylim(11.75, 12.75)
    ax.set_title(
        "Minh họa đường đi ngắn nhất trên đồ thị biển 8 hướng tại vùng vịnh Khánh Hòa",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Kinh độ (°E)")
    ax.set_ylabel("Vĩ độ (°N)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8.5)

    out_path = os.path.join(FIG_DIR, "fig2_khanhhoa_coastal_detour.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Đã xuất: {out_path}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    G, sea_mask, bounds, land = load_resources()
    plot_study_area_and_snapping(G, sea_mask, bounds, land)
    plot_coastal_detour(G, land)
