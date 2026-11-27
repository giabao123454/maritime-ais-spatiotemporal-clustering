import pickle
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.cluster import DBSCAN

GRAPH_PATH = "data/processed/maritime_graph_8dir.pkl"
FEATURES_PATH = "data/processed/ais_grid_features_2024.csv"
CLUSTERS_OUT = "data/processed/khanhhoa_baseline_clusters.csv"


def haversine_matrix(lats, lons):
    """Tính ma trận khoảng cách Haversine (km) giữa N điểm."""
    r = 6371.0088
    lat_rad = np.radians(lats)[:, None]
    lon_rad = np.radians(lons)[:, None]
    dlat = lat_rad - lat_rad.T
    dlon = lon_rad - lon_rad.T
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad) * np.cos(lat_rad.T) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * r * np.arcsin(np.sqrt(a))


def compute_truncated_maritime_dist(G, df_nodes, max_eps_km):
    """Tính ma trận thưa khoảng cách đường biển (Dijkstra cắt ngưỡng max_eps_km)."""
    n = len(df_nodes)
    node_tuples = list(zip(df_nodes["node_r"], df_nodes["node_c"]))
    node_to_idx = {node: idx for idx, node in enumerate(node_tuples)}

    # Giá trị mặc định > max_eps_km cho các cặp nằm ngoài bán kính tìm kiếm
    fill_val = max_eps_km * 2.0
    dist_mat = np.full((n, n), fill_val, dtype=np.float32)
    np.fill_diagonal(dist_mat, 0.0)

    for i, src in enumerate(node_tuples):
        lengths = nx.single_source_dijkstra_path_length(
            G, src, cutoff=max_eps_km, weight="weight"
        )
        for target, d in lengths.items():
            j = node_to_idx.get(target)
            if j is not None:
                dist_mat[i, j] = d

    return dist_mat


def run_baseline_comparison(eps_km=15.0, min_samples=15, min_fishing_hours=5.0):
    print(
        f"1. Nạp dữ liệu vùng lõi Khánh Hòa (lọc ô có tổng giờ đánh bắt >= {min_fishing_hours}h)..."
    )
    t0 = time.time()
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)["graph"]

    df = pd.read_csv(FEATURES_PATH)
    core = (
        df[(df["is_khanhhoa_core"]) & (df["total_fishing_hours"] >= min_fishing_hours)]
        .copy()
        .reset_index(drop=True)
    )
    print(
        f"   -> Số nút đưa vào phân cụm: {len(core):,} nút (chiếm {core['total_fishing_hours'].sum() / df[df['is_khanhhoa_core']]['total_fishing_hours'].sum() * 100:.2f}% giờ đánh bắt vùng lõi)"
    )

    # --- Thuật toán 1: DBSCAN chuẩn (Haversine) ---
    print(
        f"\n2. Chạy DBSCAN chuẩn (Haversine, eps={eps_km} km, min_samples={min_samples})..."
    )
    t1 = time.time()
    hav_dist = haversine_matrix(
        core["node_lat"].to_numpy(), core["node_lon"].to_numpy()
    )
    db_std = DBSCAN(eps=eps_km, min_samples=min_samples, metric="precomputed")
    core["cluster_std_dbscan"] = db_std.fit_predict(hav_dist)
    t_std = time.time() - t1

    # --- Thuật toán 2: NS-DBSCAN tĩnh (Khoảng cách đường biển 8 hướng) ---
    print(
        f"3. Chạy NS-DBSCAN tĩnh (Đường biển 8 hướng, eps={eps_km} km, min_samples={min_samples})..."
    )
    t2 = time.time()
    sea_dist = compute_truncated_maritime_dist(G, core, max_eps_km=eps_km)
    db_ns = DBSCAN(eps=eps_km, min_samples=min_samples, metric="precomputed")
    core["cluster_ns_dbscan"] = db_ns.fit_predict(sea_dist)
    t_ns = time.time() - t2

    # --- Thống kê đối chứng ---
    for name, col, elapsed in [
        ("DBSCAN chuẩn (Haversine)", "cluster_std_dbscan", t_std),
        ("NS-DBSCAN tĩnh (Đường biển)", "cluster_ns_dbscan", t_ns),
    ]:
        labels = core[col]
        n_clusters = len(set(labels)) - (1 if -1 in labels.values else 0)
        n_noise = int((labels == -1).sum())
        noise_pct = n_noise / len(core) * 100.0
        clustered_hours = core.loc[labels != -1, "total_fishing_hours"].sum()
        hours_pct = clustered_hours / core["total_fishing_hours"].sum() * 100.0
        print(f"\n   [{name}] ({elapsed:.2f}s):")
        print(f"   - Số cụm ngư trường phát hiện: {n_clusters} cụm")
        print(
            f"   - Số điểm nhiễu: {n_noise:,} nút ({noise_pct:.2f}%) | Giữ lại {hours_pct:.2f}% sản lượng giờ đánh bắt"
        )

    core.to_csv(CLUSTERS_OUT, index=False)
    print(
        f"\nHoàn tất! Đã lưu kết quả phân cụm đối chứng tại: {CLUSTERS_OUT} (Tổng thời gian: {time.time() - t0:.2f}s)"
    )


if __name__ == "__main__":
    run_baseline_comparison(eps_km=3.5, min_samples=18, min_fishing_hours=20.0)
