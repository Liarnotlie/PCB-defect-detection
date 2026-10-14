# Bộ Dữ Liệu PCB Component Detection

**LƯU Ý QUAN TRỌNG:** KHÔNG commit thư mục chứa dataset trực tiếp lên GitHub để tránh làm nặng repository.

## 1. Nguồn Dữ Liệu
- Dữ liệu gốc được lấy từ Kaggle: `vyshnavkumars/pcb-component-detection-dataset-yolo-format`
- Link: https://www.kaggle.com/datasets/vyshnavkumars/pcb-component-detection-dataset-yolo-format
- Định dạng gốc: YOLO Object Detection format.

## 2. Danh Sách Các Class (10 classes)
Hệ thống nhận diện 10 loại linh kiện trên bảng mạch PCB:
0. resistor
1. capacitor
2. connector
3. ic
4. led
5. diode
6. transistor
7. inductor
8. jumper
9. button

## 3. Cấu Trúc Dataset Sau Khi Làm Sạch (Sẵn sàng để Train)
Dataset đã qua xử lý được lưu trong thư mục `pcb_dataset_cleaned/` với cấu trúc chuẩn của YOLO:
pcb_dataset_cleaned/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
## 4. Cách Sử Dụng
Người 2 (Training) chỉ cần trỏ đường dẫn trong file cấu hình huấn luyện YOLO vào file `data.yaml` nằm trong thư mục `pcb_dataset_cleaned/`.
