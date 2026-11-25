import glob
import os

import pandas as pd

DATA_DIR = "data/raw/fleet-daily-csvs-100-v3-2024"
OUTPUT_FILE = "data/processed/KhanhHoa_NamTrungBo_2024.csv"

LAT_MIN, LAT_MAX = 10.5, 13.5
LON_MIN, LON_MAX = 108.5, 111.0


def extract_region():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print(f"Tìm thấy {len(files)} file dữ liệu ngày.")

    chunks = []
    for idx, f in enumerate(files, 1):
        df = pd.read_csv(f)
        sub = df[
            (df["cell_ll_lat"] >= LAT_MIN)
            & (df["cell_ll_lat"] <= LAT_MAX)
            & (df["cell_ll_lon"] >= LON_MIN)
            & (df["cell_ll_lon"] <= LON_MAX)
            & (df["fishing_hours"] > 0)
        ]
        chunks.append(sub)
        if idx % 50 == 0 or idx == len(files):
            print(f"Đã xử lý {idx}/{len(files)} file...")

    result = pd.concat(chunks, ignore_index=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(
        f"Hoàn tất: {len(result):,} dòng | Tổng giờ đánh bắt: {result['fishing_hours'].sum():,.2f}"
    )


if __name__ == "__main__":
    extract_region()
