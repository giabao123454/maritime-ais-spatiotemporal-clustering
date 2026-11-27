import pickle
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

GRAPH_PATH = "data/processed/maritime_graph_8dir.pkl"
GFW_CSV_PATH = "data/processed/KhanhHoa_NamTrungBo_2024.csv"
GRID_FEATURES_OUT = "data/processed/ais_grid_features_2024.csv"
MONTHLY_FEATURES_OUT = "data/processed/ais_monthly_grid_2024.csv"


def haversine_km(lat1, lon1, lat2, lon2):
    """Tính khoảng cách mặt cầu Haversine (km)."""
    r = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * np.arcsin(np.sqrt(a))


def map_cells_to_sea_nodes(unique_cells, G, sea_mask, bounds):
    """Ánh xạ toàn bộ ô lưới AIS (bao gồm 138 ô sát bờ) về nút biển hợp lệ trên đồ thị."""
    lat_min, _, lon_min, _, step = bounds
    nodes = list(G.nodes())
    node_lats = np.array([G.nodes[n]["lat"] for n in nodes])
    node_lons = np.array([G.nodes[n]["lon"] for n in nodes])
    tree = cKDTree(np.radians(np.column_stack([node_lats, node_lons])))

    mapped_r = []
    mapped_c = []
    snap_dists = []

    for lat, lon in zip(unique_cells["cell_ll_lat"], unique_cells["cell_ll_lon"]):
        r = int(round((lat - lat_min) / step))
        c = int(round((lon - lon_min) / step))

        if 0 <= r < sea_mask.shape[0] and 0 <= c < sea_mask.shape[1] and sea_mask[r, c]:
            mapped_r.append(r)
            mapped_c.append(c)
            snap_dists.append(0.0)
        else:
            _, idx = tree.query(np.radians([lat, lon]))
            nr, nc = nodes[int(idx)]
            dist = float(
                haversine_km(
                    lat, lon, G.nodes[(nr, nc)]["lat"], G.nodes[(nr, nc)]["lon"]
                )
            )
            mapped_r.append(nr)
            mapped_c.append(nc)
            snap_dists.append(dist)

    unique_cells = unique_cells.copy()
    unique_cells["node_r"] = mapped_r
    unique_cells["node_c"] = mapped_c
    unique_cells["snap_dist_km"] = snap_dists
    return unique_cells


def prepare_clustering_datasets():
    print("1. Đang nạp đồ thị biển 8 hướng và dữ liệu GFW 2024...")
    t0 = time.time()
    with open(GRAPH_PATH, "rb") as f:
        gdata = pickle.load(f)
    G, sea_mask, bounds = gdata["graph"], gdata["sea_mask"], gdata["bounds"]

    df = pd.read_csv(GFW_CSV_PATH, parse_dates=["date"])
    df["month"] = df["date"].dt.month
    print(f"   -> Đã đọc {len(df):,} bản ghi AIS ({time.time() - t0:.2f}s)")

    print("2. Đang ánh xạ các ô lưới sát bờ về nút biển gần nhất qua cKDTree...")
    unique_cells = (
        df[["cell_ll_lat", "cell_ll_lon"]].drop_duplicates().reset_index(drop=True)
    )
    cell_map = map_cells_to_sea_nodes(unique_cells, G, sea_mask, bounds)

    snapped = cell_map[cell_map["snap_dist_km"] > 0]
    print(
        f"   -> Đã ánh xạ {len(snapped)} ô sát bờ | Khoảng cách dịch chuyển TB: {snapped['snap_dist_km'].mean():.2f} km (Tối đa: {snapped['snap_dist_km'].max():.2f} km)"
    )

    # Gắn tọa độ nút đồ thị chuẩn hóa vào toàn bộ 716.885 dòng
    df = df.merge(
        cell_map[["cell_ll_lat", "cell_ll_lon", "node_r", "node_c"]],
        on=["cell_ll_lat", "cell_ll_lon"],
        how="left",
    )
    df["node_lat"] = df["node_r"].map(lambda r: bounds[0] + r * bounds[4]).round(2)
    df["node_lon"] = df["node_c"].map(lambda c: bounds[2] + c * bounds[4]).round(2)

    # Xác định cột số tàu hoạt động nếu tồn tại trong GFW v3.0
    has_mmsi = "mmsi_present" in df.columns

    print("3. Đang tổng hợp đặc trưng không gian - thời gian theo từng nút biển...")
    agg_dict = {
        "total_fishing_hours": ("fishing_hours", "sum"),
        "mean_daily_hours": ("fishing_hours", "mean"),
        "active_days": ("date", "nunique"),
        "active_months": ("month", "nunique"),
        "record_count": ("fishing_hours", "count"),
    }
    if has_mmsi:
        agg_dict["total_mmsi_present"] = ("mmsi_present", "sum")

    grid_features = (
        df.groupby(["node_r", "node_c", "node_lat", "node_lon"], as_index=False)
        .agg(**agg_dict)
        .sort_values("total_fishing_hours", ascending=False)
        .reset_index(drop=True)
    )
    # Đánh dấu vùng lõi đánh giá Khánh Hòa (11.3°N - 13.2°N, < 110.0°E)
    grid_features["is_khanhhoa_core"] = (
        (grid_features["node_lat"] >= 11.3)
        & (grid_features["node_lat"] <= 13.2)
        & (grid_features["node_lon"] < 110.0)
    )

    monthly_agg = {
        "fishing_hours": ("fishing_hours", "sum"),
        "active_days": ("date", "nunique"),
    }
    if has_mmsi:
        monthly_agg["mmsi_present"] = ("mmsi_present", "sum")

    monthly_features = df.groupby(
        ["month", "node_r", "node_c", "node_lat", "node_lon"], as_index=False
    ).agg(**monthly_agg)

    grid_features.to_csv(GRID_FEATURES_OUT, index=False)
    monthly_features.to_csv(MONTHLY_FEATURES_OUT, index=False)

    core_df = grid_features[grid_features["is_khanhhoa_core"]]
    print(
        f"   -> Tổng nút biển có hoạt động đánh bắt (sau gộp mép bờ): {len(grid_features):,} nút"
    )
    print(
        f"   -> Vùng lõi Khánh Hòa (11.3°N–13.2°N, <110°E): {len(core_df):,} nút | Chiếm {core_df['total_fishing_hours'].sum() / grid_features['total_fishing_hours'].sum() * 100:.2f}% tổng giờ đánh bắt"
    )
    print(f"   -> Đã xuất dữ liệu tổng hợp: {GRID_FEATURES_OUT}")
    print(f"   -> Đã xuất dữ liệu theo tháng: {MONTHLY_FEATURES_OUT}")


if __name__ == "__main__":
    prepare_clustering_datasets()
