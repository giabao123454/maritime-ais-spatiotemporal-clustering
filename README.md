# Maritime AIS Spatio-Temporal Clustering

**Đề tài:** Cải tiến thuật toán phân cụm dựa trên mật độ trong không gian mạng đường biển cho dữ liệu nỗ lực đánh bắt AIS vùng biển Nam Trung Bộ (trọng tâm Khánh Hòa).

* **Sinh viên thực hiện:** Nguyễn Ngọc Gia Bảo (MSSV: 65130193 – Lớp: 65.CNTT-1, Trường Đại học Nha Trang)
* **Giảng viên hướng dẫn:** TS. Nguyễn Thủy Đoan Trang
* **Phạm vi nghiên cứu:** Khung tọa độ $10.5^\circ\text{N} - 13.5^\circ\text{N}$, $108.5^\circ\text{E} - 111.0^\circ\text{E}$ (Vùng lõi đánh giá: $11.3^\circ\text{N} - 13.2^\circ\text{N}$, $< 110^\circ\text{E}$).

## 1. Cấu trúc thư mục
```text
├── data/                  # Dữ liệu cục bộ (bị loại trừ khỏi Git qua .gitignore)
├── docs/                  # Đề xuất đề tài Giai đoạn 1, minh chứng SJR và báo cáo tuần
├── scripts/               # Mã nguồn trích xuất & tiền xử lý dữ liệu thô GFW/OSM
├── src/                   # Lõi thuật toán (đồ thị biển, baselines, thuật toán cải tiến)
├── app/                   # Dashboard bản đồ tương tác (Streamlit + Folium)
├── .gitignore
├── requirements.txt
└── README.md