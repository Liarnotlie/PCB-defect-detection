# Tài Liệu Dữ Liệu (Dataset Documentation)

Tài liệu này tổng hợp các phân tích thăm dò dữ liệu (EDA) và các bước tiền xử lý đã được thực hiện để chuẩn bị dataset cho mô hình YOLO Object Detection.

## 1. Thống Kê Tổng Quan (Dataset Statistics)
- **Tổng số ảnh hợp lệ:** 460 ảnh
- **Tổng số class:** 10
- **Tỉ lệ phân chia (Split Ratio):** 80% Train / 10% Val / 10% Test
  - **Train set:** 368 ảnh
  - **Validation set:** 46 ảnh
  - **Test set:** 46 ảnh

## 2. Các Bước Tiền Xử Lý (Data Processing)
Mã nguồn xử lý: `src/preprocessing.py`

Quá trình làm sạch dữ liệu đã thực hiện các bước sau:
1. **Quét và đối chiếu (Mapping):** Bắt cặp tự động file ảnh (`.jpg`, `.png`) và file nhãn (`.txt`).
2. **Kiểm tra tính hợp lệ:** Bỏ qua các nhãn có tọa độ sai format hoặc class_id không nằm trong danh sách.
3. **Chuẩn hóa Bounding Box:** Phát hiện và tự động sửa **92 bounding box** bị lỗi tràn viền ảnh. Các hộp bị tràn đã được ép lại vừa khít với giới hạn [0.0, 1.0] của format YOLO.
4. **Lọc ảnh rác:** Loại bỏ hoàn toàn các ảnh bị lỗi không thể mở bằng OpenCV.
5. **Chia tập dữ liệu:** Trộn ngẫu nhiên (shuffle) với `seed=42` và chia tập dữ liệu thành Train/Val/Test, xuất ra kèm file `data.yaml` tự động hóa.

## 3. Phân Tích Khám Phá Dữ Liệu (EDA)
Mã nguồn phân tích: `notebooks/eda.ipynb`

- Các linh kiện trên PCB có kích thước rất đa dạng, từ rất nhỏ (như điện trở - resistor) đến lớn (như IC, connector).
- Cần lưu ý việc mất cân bằng dữ liệu (Imbalanced Data) giữa các class phổ biến và các class hiếm khi huấn luyện mô hình.
- Các biểu đồ chi tiết về phân bố class, kích thước bounding box và số lượng object trên mỗi ảnh có thể xem trực tiếp khi chạy file `eda.ipynb`.