"""Kiểm tra nhanh dữ liệu GFW và vẽ bản đồ các ô lưới đánh bắt.

Cách chạy (PowerShell, trong thư mục D:/GFW_Data):
    pip install pandas matplotlib
    python scripts/check_gfw.py
Kết quả: in thống kê ra màn hình và lưu ảnh gfw_map.png cùng thư mục.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

path = (
    sys.argv[1] if len(sys.argv) > 1 else "data/processed/KhanhHoa_NamTrungBo_2024.csv"
)
df = pd.read_csv(path, parse_dates=["date"])

print("Số dòng:", len(df))
print("Khoảng ngày:", df["date"].min().date(), "->", df["date"].max().date())
print("Vĩ độ ô   min/max:", df["cell_ll_lat"].min(), df["cell_ll_lat"].max())
print("Kinh độ ô min/max:", df["cell_ll_lon"].min(), df["cell_ll_lon"].max())

cells = df.groupby(["cell_ll_lat", "cell_ll_lon"], as_index=False)[
    "fishing_hours"
].sum()
print("Số ô lưới duy nhất:", len(cells))
print("Tổng cột hours        :", round(df["hours"].sum(), 2))
print("Tổng cột fishing_hours:", round(df["fishing_hours"].sum(), 2))

print("\nSố dòng theo cờ (flag), 10 giá trị đầu:")
print(df["flag"].value_counts().head(10))
print("\nSố dòng theo tháng:")
print(df.groupby(df["date"].dt.month).size())

pos = cells[cells["fishing_hours"] > 0]
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(
    pos["cell_ll_lon"],
    pos["cell_ll_lat"],
    c=pos["fishing_hours"],
    s=3,
    cmap="viridis",
    norm=LogNorm(),
)
fig.colorbar(sc, ax=ax, label="Tổng giờ đánh bắt (thang log)")
ax.set_xlabel("Kinh độ")
ax.set_ylabel("Vĩ độ")
ax.set_title("Các ô lưới đánh bắt biểu kiến")
ax.set_aspect("equal")
fig.savefig("gfw_map.png", dpi=150, bbox_inches="tight")
print("\nĐã lưu gfw_map.png")
