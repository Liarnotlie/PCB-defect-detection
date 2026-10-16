# Báo Cáo Đóng Gói, Inference & Triển Khai (PCB Component Detection)

Tài liệu này mô tả toàn bộ công việc của Người 3 trong dự án: xây dựng Inference Pipeline, ứng dụng web Streamlit, chuẩn bị tài nguyên demo và hướng dẫn triển khai hệ thống.

---

## 1. Inference Pipeline (src/inference.py)

Mã nguồn inference được tích hợp trực tiếp vào `app/app.py` thông qua lớp `YOLO` của thư viện Ultralytics. Luồng xử lý như sau:

```
Input Image
     |
     v
Preprocessing  (PIL.Image -> RGB, resize nội bộ do YOLO xử lý tại 640x640)
     |
     v
YOLO Object Detection  (model.predict, conf threshold, IoU/NMS)
     |
     v
Bounding Boxes + Class IDs + Confidence Scores
     |
     v
Post-processing  (lọc box, đếm số lượng, tổng hợp theo class)
     |
     v
Application Output  (ảnh annotated + bảng kết quả + tóm tắt)
```

### 1.1. Load Model

Model được nạp một lần duy nhất và tái sử dụng giữa các lần chạy Streamlit nhờ decorator `@st.cache_resource`:

```python
@st.cache_resource(show_spinner=False)
def load_model(path: str) -> YOLO:
    return YOLO(path)
```

- Đường dẫn mặc định: `model/best.pt`
- Framework: Ultralytics YOLOv8

### 1.2. Chạy Inference

```python
result = model.predict(image, conf=confidence, iou=iou, verbose=False)[0]
```

- `conf`: Ngưỡng tin cậy (mặc định 0.25), điều chỉnh qua thanh trượt sidebar.
- `iou`: Ngưỡng IoU cho Non-Maximum Suppression (mặc định 0.45).
- `verbose=False`: Tắt log console để giữ giao diện sạch.

### 1.3. Lấy Kết Quả

Sau khi chạy predict, kết quả được trích xuất từ `result.boxes`:

| Trường | Mô tả |
|---|---|
| `box.cls` | Class ID (số nguyên, ánh xạ qua `result.names`) |
| `box.conf` | Độ tin cậy (float, 0.0 – 1.0) |
| `box.xyxy` | Tọa độ bounding box dạng `(x1, y1, x2, y2)` theo pixel |

---

## 2. Post-processing

### 2.1. Các Bước Xử Lý

1. **Lọc theo ngưỡng tin cậy (Confidence Threshold):** Ultralytics tự động lọc tại bước `model.predict`. Người dùng điều chỉnh ngưỡng này qua slider sidebar (0.05 – 0.95).
2. **Lọc box không hợp lệ:** Chỉ xử lý khi `result.boxes is not None` và box có đủ thông tin class, conf, xyxy.
3. **Tính kích thước box:** `w = x2 - x1`, `h = y2 - y1` — dùng để hiển thị kích thước linh kiện.
4. **Đếm số lượng theo class:** Sử dụng `collections.Counter` trên danh sách tên linh kiện:
   ```python
   counter = Counter(d["Linh kiện"] for d in detections)
   ```
5. **Tổng hợp kết quả:** Kết quả được trình bày theo ba dạng:
   - Badge chips tóm tắt số lượng theo class.
   - Confidence bar có màu sắc (xanh ≥ 70%, vàng ≥ 45%, đỏ < 45%).
   - Bảng chi tiết đầy đủ với tên, độ tin cậy, kích thước và tọa độ box.

### 2.2. Ví Dụ Đầu Ra Tổng Hợp

```
Capacitor : 12
Resistor  :  8
Connector :  4
IC        :  2
LED       :  1
```

---

## 3. Ứng Dụng Web (app/app.py)

Ứng dụng được xây dựng bằng **Streamlit** với giao diện đa trang, chủ đề tối (dark theme) tuỳ chỉnh hoàn toàn bằng CSS nội tuyến.

### 3.1. Cấu Trúc Đa Trang

| Trang | Chức năng |
|---|---|
| Trang chủ | Giới thiệu dự án, thống kê tổng quan, catalog 10 loại linh kiện, pipeline |
| Nhận diện | Upload ảnh, chạy inference, hiển thị kết quả, tải ảnh xuống |
| Hiệu suất mô hình | Cấu hình training, bảng metrics, đường cong huấn luyện, confusion matrix |
| Gallery mẫu | Lưới 3 cột các ảnh kết quả mẫu từ tập test |

### 3.2. Tính Năng Trang Nhận Diện

- Upload ảnh định dạng JPG, JPEG, PNG, BMP, WEBP.
- Hiển thị đồng thời ảnh gốc và ảnh đã vẽ bounding box.
- 4 thẻ số liệu nhanh: tổng linh kiện, số loại khác nhau, thời gian inference (ms), ngưỡng tin cậy.
- Badge chips tóm tắt số lượng theo từng class.
- Confidence bars trực quan hoá độ tin cậy từng detection.
- Bảng chi tiết đầy đủ có thể ẩn/hiện qua toggle sidebar.
- Nút tải ảnh kết quả dạng PNG.
- Toggle hiển thị/ẩn ảnh gốc và bảng chi tiết ngay trên sidebar.

### 3.3. Dependencies

```
streamlit
ultralytics
pandas
```

Cài đặt:
```bash
pip install -r requirements.txt
```

---

## 4. Sơ Đồ Kiến Trúc (Architecture Diagram)

```
+---------------------------+
|       Input Image         |   (JPG / PNG / BMP / WEBP)
+---------------------------+
              |
              v
+---------------------------+
|       Preprocessing       |   PIL.Image.open() -> convert("RGB")
|                           |   (resize nội bộ tại 640x640 bởi YOLO)
+---------------------------+
              |
              v
+---------------------------+
|   YOLO Object Detection   |   model.predict(conf, iou, verbose=False)
|   (YOLOv8n - best.pt)     |   Ultralytics inference engine
+---------------------------+
              |
              v
+---------------------------+
|  Bounding Box + Class +   |   result.boxes: cls, conf, xyxy
|  Confidence Score         |
+---------------------------+
              |
              v
+---------------------------+
|      Post-processing      |   Lọc, đếm, tổng hợp theo class
|                           |   result.plot() -> annotated image
+---------------------------+
              |
              v
+---------------------------+
|    Streamlit Application  |   Hiển thị ảnh, bảng, badge, bars
|    (app/app.py)           |   Nút download kết quả
+---------------------------+
              |
              v
+---------------------------+
|          Result           |   Ảnh annotated + bảng chi tiết
|                           |   + tóm tắt số lượng theo class
+---------------------------+
```

Sơ đồ kiến trúc dạng hình ảnh lưu tại: `assest/architecture/` _(xem hướng dẫn thêm ảnh ở Mục 5)_.

---

## 5. Demo (assest/demo/)

### 5.1. Luồng Demo

```
Input PCB Image
       |
       v
Upload lên giao diện web
       |
       v
YOLO Detection (YOLOv8n)
       |
       v
Bounding Boxes vẽ trực tiếp lên ảnh
       |
       v
Component Classification (10 classes)
       |
       v
Component Count (đếm số lượng từng loại)
       |
       v
Hiển thị kết quả: ảnh + bảng + badge
```

### 5.2. Tài Nguyên Demo

Các ảnh mẫu kết quả inference trên tập test đang được lưu tại:

- `assest/pred/prediction_01.jpg` đến `prediction_20.jpg` — 20 ảnh kết quả nhận diện.
- `assest/pred/test_predictions_sample.png` — ảnh tổng hợp toàn bộ mẫu test.
- `assest/graph/comprehensive_training_metrics.png` — đường cong training.
- `assest/graph/confusion_matrix.png` — ma trận nhầm lẫn.

### 5.3. Hướng Dẫn Thêm Ảnh Demo Vào Thư Mục

Để bổ sung ảnh demo (ảnh input, ảnh output, screenshot giao diện), thực hiện các bước sau:

**Bước 1 — Tạo thư mục nếu chưa có:**
```bash
mkdir assest\demo
mkdir assest\architecture
```

**Bước 2 — Đặt ảnh vào đúng thư mục:**

| Loại ảnh | Thư mục đích | Tên file gợi ý |
|---|---|---|
| Ảnh PCB gốc (input) | `assest/demo/` | `demo_input_01.jpg` |
| Ảnh kết quả nhận diện (output) | `assest/demo/` | `demo_output_01.jpg` |
| Screenshot giao diện web | `assest/demo/` | `screenshot_home.png`, `screenshot_detect.png` |
| Sơ đồ kiến trúc | `assest/architecture/` | `architecture_diagram.png` |

**Bước 3 — Chụp screenshot giao diện Streamlit:**
1. Chạy ứng dụng: `streamlit run app/app.py`
2. Mở trình duyệt tại `http://localhost:8501`
3. Chụp từng trang bằng phím `Windows + Shift + S` (Windows Snipping Tool) hoặc công cụ chụp màn hình của hệ điều hành.
4. Lưu ảnh vào `assest/demo/`.

**Bước 4 — Commit ảnh vào repository:**
```bash
git add assest/demo/ assest/architecture/
git commit -m "docs(demo): add demo screenshots and architecture diagram"
```

---

## 6. Hướng Dẫn Triển Khai (Deployment Guide)

### 6.1. Yêu Cầu Phần Cứng / Phần Mềm

| Yêu cầu | Tối thiểu |
|---|---|
| Hệ điều hành | Windows 10/11, macOS 12+, Ubuntu 20.04+ |
| Python | 3.9 trở lên |
| RAM | 4 GB (khuyến nghị 8 GB) |
| Dung lượng ổ cứng | 2 GB trống (cho model + dependencies) |
| GPU | Không bắt buộc — CPU đủ để chạy YOLOv8n |

### 6.2. Cài Đặt Môi Trường

**Tạo và kích hoạt môi trường ảo Python:**

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt (Windows)
.venv\Scripts\activate

# Kích hoạt (macOS / Linux)
source .venv/bin/activate
```

**Cài đặt dependencies:**

```bash
pip install -r requirements.txt
```

Nội dung `requirements.txt`:
```
streamlit
ultralytics
pandas
```

### 6.3. Chuẩn Bị Model

Đảm bảo file trọng số đã được đặt đúng vị trí:

```
PCB-defect-detection/
└── model/
    └── best.pt       <-- file này phải tồn tại
```

Nếu chưa có file `best.pt`, lấy từ kết quả training tại đường dẫn Ultralytics mặc định:
`runs/detect/train/weights/best.pt`, sau đó sao chép vào `model/`.

### 6.4. Chạy Ứng Dụng Local

Từ thư mục gốc của repository:

```bash
streamlit run app/app.py
```

Trình duyệt tự động mở tại `http://localhost:8501`. Nếu không tự mở, truy cập thủ công địa chỉ trên.

### 6.5. Chạy Inference Đơn Lẻ (Không Qua Giao Diện)

Có thể gọi model trực tiếp bằng Python để kiểm tra nhanh:

```python
from ultralytics import YOLO
from PIL import Image

model = YOLO("model/best.pt")
image = Image.open("duong/dan/anh_pcb.jpg").convert("RGB")
results = model.predict(image, conf=0.25, iou=0.45, verbose=False)

result = results[0]
for box in result.boxes:
    cls_id = int(box.cls.item())
    conf   = float(box.conf.item())
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"{result.names[cls_id]}: {conf:.2%}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
```

### 6.6. Triển Khai Trên Hugging Face Spaces

Nếu muốn chia sẻ ứng dụng lên Hugging Face Spaces (không cần Docker):

```
Trained Model (model/best.pt)
       |
       v
Inference Code (app/app.py)
       |
       v
Streamlit Application
       |
       v
Hugging Face Spaces (SDK: streamlit)
       |
       v
Web Demo (public URL)
```

**Các bước thực hiện:**

1. Tạo tài khoản tại [huggingface.co](https://huggingface.co).
2. Tạo Space mới, chọn SDK là **Streamlit**.
3. Clone repository Space về máy:
   ```bash
   git clone https://huggingface.co/spaces/<username>/<space-name>
   ```
4. Sao chép toàn bộ mã nguồn vào thư mục Space.
5. Đổi tên file chính thành `app.py` ở thư mục gốc (hoặc khai báo trong `README.md` của Space).
6. Thêm `requirements.txt` vào thư mục gốc Space.
7. Upload `model/best.pt` — nếu file quá lớn (> 100 MB), sử dụng **Hugging Face Git LFS**:
   ```bash
   git lfs install
   git lfs track "*.pt"
   git add .gitattributes
   ```
8. Commit và push:
   ```bash
   git add .
   git commit -m "deploy: initial Streamlit app"
   git push
   ```
9. Space tự động build và deploy. Theo dõi log tại tab **Logs** trên giao diện web của Space.

### 6.7. Các Lỗi Thường Gặp

| Lỗi | Nguyên nhân | Cách khắc phục |
|---|---|---|
| `FileNotFoundError: model/best.pt` | Thiếu file model | Sao chép `best.pt` vào thư mục `model/` |
| `ModuleNotFoundError: No module named 'ultralytics'` | Chưa cài dependencies | Chạy `pip install -r requirements.txt` |
| `OSError: cannot identify image file` | File ảnh bị lỗi hoặc sai định dạng | Thử ảnh khác hoặc convert sang JPG/PNG |
| Streamlit không mở trình duyệt | Tường lửa hoặc port bị chặn | Truy cập thủ công `http://localhost:8501` |
| Inference quá chậm | CPU cũ hoặc ảnh quá lớn | Giảm độ phân giải ảnh đầu vào trước khi upload |
| Không phát hiện linh kiện nào | Ngưỡng tin cậy quá cao | Kéo slider **Ngưỡng tin cậy** xuống thấp hơn (ví dụ 0.10) |

---

## 7. Danh Sách File Phụ Trách

| File / Thư mục | Mô tả |
|---|---|
| `app/app.py` | Toàn bộ ứng dụng web Streamlit (4 trang) + inference pipeline |
| `assest/pred/` | 20 ảnh kết quả nhận diện mẫu trên tập test |
| `assest/graph/` | Đường cong training + confusion matrix |
| `assest/demo/` | Ảnh demo, screenshot giao diện _(cần bổ sung thủ công)_ |
| `assest/architecture/` | Sơ đồ kiến trúc hệ thống _(cần bổ sung thủ công)_ |
| `docs/deployment.md` | Tài liệu này |
| `requirements.txt` | Danh sách dependencies toàn dự án |
| `README.md` | Tổng hợp hướng dẫn sử dụng (cả nhóm hoàn thiện) |

