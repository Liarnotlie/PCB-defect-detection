"""Simple local web interface for the PCB component detector."""

from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "best.pt"


st.set_page_config(page_title="PCB Component Detection", page_icon="🔎", layout="wide")


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    """Load the trained model once and reuse it between Streamlit reruns."""
    return YOLO(model_path)


st.title("Nhận diện linh kiện trên mạch PCB")
st.write(
    "Tải một ảnh PCB lên để mô hình YOLO phát hiện và phân loại linh kiện. "
    "Mô hình chạy trực tiếp trên máy đang mở ứng dụng."
)

if not MODEL_PATH.is_file():
    st.error(f"Không tìm thấy model tại `{MODEL_PATH}`. Hãy kiểm tra file `model/best.pt`.")
    st.stop()

with st.sidebar:
    st.header("Cấu hình nhận diện")
    confidence = st.slider("Ngưỡng tin cậy", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    iou = st.slider("Ngưỡng IoU", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

uploaded_file = st.file_uploader(
    "Chọn ảnh PCB từ máy tính",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Các định dạng ảnh phổ biến được hỗ trợ.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    model = load_model(str(MODEL_PATH))

    with st.spinner("Đang nhận diện linh kiện..."):
        result = model.predict(image, conf=confidence, iou=iou, verbose=False)[0]

    annotated_image = Image.fromarray(result.plot()[..., ::-1])
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(
                {
                    "Linh kiện": result.names[class_id],
                    "Độ tin cậy": round(float(box.conf.item()), 3),
                    "Tọa độ hộp (x1, y1, x2, y2)": f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})",
                }
            )

    left, right = st.columns(2)
    with left:
        st.subheader("Ảnh gốc")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Kết quả nhận diện")
        st.image(annotated_image, use_container_width=True)

    st.subheader(f"Linh kiện phát hiện được: {len(detections)}")
    if detections:
        st.dataframe(detections, hide_index=True, use_container_width=True)
    else:
        st.info("Không tìm thấy linh kiện nào trên ảnh với ngưỡng tin cậy hiện tại.")
else:
    st.info("Tải ảnh lên để bắt đầu nhận diện.")
