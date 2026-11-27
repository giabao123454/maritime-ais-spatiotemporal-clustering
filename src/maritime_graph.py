import os
import pickle
import time

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, box

# Khung tọa độ nghiên cứu (Khánh Hòa - Nam Trung Bộ)
LAT_MIN, LAT_MAX = 10.5, 13.5
LON_MIN, LON_MAX = 108.5, 111.0
STEP = 0.01  # Độ phân giải 0.01 độ khớp với GFW v3.0

OSM_SHP_PATH = "data/raw/osm_land_polygons/land_polygons.shp"
GFW_CSV_PATH = "data/processed/KhanhHoa_NamTrungBo_2024.csv"
CLIPPED_LAND_PATH = "data/processed/khanhhoa_namtrungbo_land.gpkg"
GRAPH_OUTPUT_PATH = "data/processed/maritime_graph_8dir.pkl"


def haversine_km(lat1, lon1, lat2, lon2):
    """Tính khoảng cách mặt cầu Haversine (km) giữa hai tọa độ."""
    r = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * np.arcsin(np.sqrt(a))


def clip_osm_land():
    """Đọc và cắt riêng đa giác đất liền/hải đảo trong khung tọa độ nghiên cứu."""
    print("1. Đang nạp và cắt dữ liệu đường bờ biển OSM theo bbox...")
    t0 = time.time()
    bbox_geom = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    land_gdf = gpd.read_file(OSM_SHP_PATH, bbox=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX))
    land_clipped = gpd.clip(land_gdf, bbox_geom)

    os.makedirs(os.path.dirname(CLIPPED_LAND_PATH), exist_ok=True)
    land_clipped.to_file(CLIPPED_LAND_PATH, driver="GPKG")
    print(
        f"   -> Đã cắt {len(land_clipped)} đa giác đất liền/đảo trong {time.time() - t0:.2f}s"
    )
    print(f"   -> Đã lưu bản đồ vùng nghiên cứu tại: {CLIPPED_LAND_PATH}")
    return land_clipped


def build_maritime_grid_graph(land_clipped):
    """Xây dựng đồ thị lưới biển 8 hướng (chống cắt góc qua đất liền)."""
    print("2. Đang khởi tạo lưới 0.01° và lọc bỏ các ô đất liền/đảo...")
    t0 = time.time()

    n_lat = int(round((LAT_MAX - LAT_MIN) / STEP)) + 1
    n_lon = int(round((LON_MAX - LON_MIN) / STEP)) + 1

    lats = np.round(LAT_MIN + np.arange(n_lat) * STEP, 2)
    lons = np.round(LON_MIN + np.arange(n_lon) * STEP, 2)

    grid_r, grid_c = np.meshgrid(np.arange(n_lat), np.arange(n_lon), indexing="ij")
    flat_r = grid_r.ravel()
    flat_c = grid_c.ravel()
    flat_lat = lats[flat_r]
    flat_lon = lons[flat_c]

    # Tâm ô lưới 0.01° (cell_ll_lat + 0.005, cell_ll_lon + 0.005)
    centers = gpd.GeoDataFrame(
        {"r": flat_r, "c": flat_c, "lat": flat_lat, "lon": flat_lon},
        geometry=[
            Point(lon + 0.005, lat + 0.005) for lat, lon in zip(flat_lat, flat_lon)
        ],
        crs="EPSG:4326",
    )

    # Xác định các tâm ô rơi vào đa giác đất liền/đảo
    land_union = land_clipped.union_all()
    is_land = centers.geometry.within(land_union).to_numpy()
    sea_mask = ~is_land.reshape((n_lat, n_lon))

    total_cells = n_lat * n_lon
    sea_cells = int(sea_mask.sum())
    land_cells = total_cells - sea_cells
    print(
        f"   -> Tổng số ô lưới: {total_cells:,} | Ô biển: {sea_cells:,} | Ô đất liền/đảo: {land_cells:,}"
    )

    # Dựng đồ thị NetworkX 8 hướng
    G = nx.Graph()
    for idx in np.where(~is_land)[0]:
        r, c = int(flat_r[idx]), int(flat_c[idx])
        G.add_node((r, c), lat=float(flat_lat[idx]), lon=float(flat_lon[idx]))

    # 4 hướng thẳng + 4 hướng chéo (chỉ nối chéo khi 2 ô kề cạnh đều là biển để không cắt qua mũi đất)
    cardinal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diagonal_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(n_lat):
        for c in range(n_lon):
            if not sea_mask[r, c]:
                continue
            lat1, lon1 = lats[r], lons[c]

            for dr, dc in cardinal_dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_lat and 0 <= nc < n_lon and sea_mask[nr, nc]:
                    if (r, c) < (nr, nc):
                        dist = float(haversine_km(lat1, lon1, lats[nr], lons[nc]))
                        G.add_edge((r, c), (nr, nc), weight=dist)

            for dr, dc in diagonal_dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_lat and 0 <= nc < n_lon and sea_mask[nr, nc]:
                    if sea_mask[r + dr, c] and sea_mask[r, c + dc]:
                        if (r, c) < (nr, nc):
                            dist = float(haversine_km(lat1, lon1, lats[nr], lons[nc]))
                            G.add_edge((r, c), (nr, nc), weight=dist)

    print(
        f"   -> Đồ thị hoàn tất: {G.number_of_nodes():,} nút, {G.number_of_edges():,} cạnh ({time.time() - t0:.2f}s)"
    )
    return G, sea_mask


def verify_gfw_alignment(sea_mask):
    """Đối chiếu 45.082 ô lưới AIS thực tế với mặt nạ biển vừa dựng."""
    print("3. Đang đối chiếu độ khớp với dữ liệu GFW 2024...")
    df = pd.read_csv(
        GFW_CSV_PATH, usecols=["cell_ll_lat", "cell_ll_lon", "fishing_hours"]
    )
    unique_cells = df.groupby(["cell_ll_lat", "cell_ll_lon"], as_index=False)[
        "fishing_hours"
    ].sum()

    r_idx = np.round((unique_cells["cell_ll_lat"] - LAT_MIN) / STEP).astype(int)
    c_idx = np.round((unique_cells["cell_ll_lon"] - LON_MIN) / STEP).astype(int)

    in_sea = sea_mask[r_idx, c_idx]
    n_in_sea = int(in_sea.sum())
    n_on_land = len(unique_cells) - n_in_sea
    hours_in_sea = unique_cells.loc[in_sea, "fishing_hours"].sum()
    total_hours = unique_cells["fishing_hours"].sum()

    print(f"   -> Tổng ô AIS GFW: {len(unique_cells):,} ô")
    print(
        f"   -> Khớp trên biển: {n_in_sea:,} ô ({n_in_sea / len(unique_cells) * 100:.2f}%) | Chiếm {hours_in_sea / total_hours * 100:.2f}% giờ đánh bắt"
    )
    print(
        f"   -> Dính sát mép bờ OSM (cần ánh xạ về nút biển gần nhất): {n_on_land:,} ô"
    )


def build_sea_kdtree(G):
    """Tạo cây tìm kiếm không gian để ánh xạ các điểm sát bờ về nút biển gần nhất."""
    nodes = list(G.nodes())
    coords_rad = np.radians([(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in nodes])
    return nodes, cKDTree(coords_rad)


def snap_to_sea_node(lat, lon, nodes, tree):
    """Tìm nút biển gần nhất trên đồ thị cho một cặp tọa độ (lat, lon)."""
    _, idx = tree.query(np.radians([lat, lon]))
    return nodes[int(idx)]


def benchmark_coastal_detour(G):
    """Đo độ lệch khoảng cách đường biển vs đường chim bay tại các vịnh/bán đảo Khánh Hòa."""
    print("4. Thử nghiệm đo khoảng cách đường biển né bán đảo/đảo:")
    nodes, tree = build_sea_kdtree(G)

    test_pairs = [
        (
            "Vịnh Vân Phong <-> Đầm Nha Phu (né bán đảo Hòn Hèo)",
            (12.64, 109.32),
            (12.40, 109.19),
        ),
        (
            "Vịnh Nha Trang <-> Vịnh Cam Ranh (né bán đảo Cam Ranh)",
            (12.21, 109.23),
            (11.88, 109.20),
        ),
        ("Phía Tây <-> Phía Đông đảo Phú Quý", (10.53, 108.88), (10.53, 109.02)),
    ]

    for name, (lat_a, lon_a), (lat_b, lon_b) in test_pairs:
        node_a = snap_to_sea_node(lat_a, lon_a, nodes, tree)
        node_b = snap_to_sea_node(lat_b, lon_b, nodes, tree)

        d_hav = haversine_km(
            G.nodes[node_a]["lat"],
            G.nodes[node_a]["lon"],
            G.nodes[node_b]["lat"],
            G.nodes[node_b]["lon"],
        )
        d_sea = nx.shortest_path_length(
            G, source=node_a, target=node_b, weight="weight"
        )
        detour = (d_sea / d_hav - 1.0) * 100.0
        print(f"   * {name}:")
        print(
            f"     Chim bay = {d_hav:.2f} km | Đường biển = {d_sea:.2f} km (Vòng xa hơn +{detour:.1f}%)"
        )


if __name__ == "__main__":
    land = clip_osm_land()
    G, sea_mask = build_maritime_grid_graph(land)
    verify_gfw_alignment(sea_mask)
    benchmark_coastal_detour(G)

    with open(GRAPH_OUTPUT_PATH, "wb") as f:
        pickle.dump(
            {
                "graph": G,
                "sea_mask": sea_mask,
                "bounds": (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, STEP),
            },
            f,
        )
    print(f"\nHoàn tất! Đã lưu đồ thị biển 8 hướng tại: {GRAPH_OUTPUT_PATH}")
