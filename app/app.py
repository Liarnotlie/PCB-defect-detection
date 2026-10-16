"""
PCB Component Detection — Streamlit Web Application
Navigation: horizontal tabs | Home | Detect | Performance | Gallery
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from collections import Counter

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ─────────────────────── paths ───────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "model"   / "best.pt"
ASSETS       = PROJECT_ROOT / "assest"
GRAPH_DIR    = ASSETS / "graph"
PRED_DIR     = ASSETS / "pred"
METRICS_CSV  = ASSETS / "evaluation_metrics.csv"

# ─────────────────────── page config ─────────────────
st.set_page_config(
    page_title="PCB Component Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",   # sidebar bị thu nhỏ mặc định
)

# ─────────────────────── global CSS ──────────────────
st.markdown(
    """
    <style>
    /* ═══════════════════════════════════════════════
       1. Xử lý header / toolbar mặc định của Streamlit
          — Đây là nguyên nhân gây viền trắng ở trên cùng
    ═══════════════════════════════════════════════ */
    html, body {
        background: #0d1117 !important;
    }

    /* Header thanh ngang trên cùng của Streamlit */
    header[data-testid="stHeader"] {
        background: #0d1117 !important;
        border-bottom: none !important;
        box-shadow: none !important;
    }

    /* Toolbar (nút deploy, menu 3 chấm) */
    [data-testid="stToolbar"] {
        background: #0d1117 !important;
    }

    /* Ẩn hamburger menu và nút collapse sidebar */
    #MainMenu { visibility: hidden !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }

    /* Sidebar — ẩn hoàn toàn, không dùng nữa */
    [data-testid="stSidebar"] { display: none !important; }

    /* Kéo content lên sát đầu trang (xoá padding-top thừa) */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }

    /* ═══════════════════════════════════════════════
       2. Base — nền toàn trang
    ═══════════════════════════════════════════════ */
    [data-testid="stAppViewContainer"] {
        background: #0d1117 !important;
        color: #e6edf3;
    }
    [data-testid="stAppViewContainer"] > section {
        background: #0d1117 !important;
    }

    /* ═══════════════════════════════════════════════
       3. Tabs — thanh điều hướng ngang
    ═══════════════════════════════════════════════ */
    [data-testid="stTabs"] {
        background: #0d1117;
        border-bottom: 1px solid #21262d;
        margin-bottom: 24px;
    }
    [data-testid="stTabs"] > div > div {
        gap: 4px;
    }
    /* Tab button chưa chọn */
    button[data-baseweb="tab"] {
        background: transparent !important;
        color: #8b949e !important;
        font-size: .93rem !important;
        font-weight: 500 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px !important;
        border: none !important;
        transition: color .18s, background .18s !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #e6edf3 !important;
        background: #161b22 !important;
    }
    /* Tab button đang chọn */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #58a6ff !important;
        background: #161b22 !important;
        border-bottom: 2px solid #1f6feb !important;
        font-weight: 700 !important;
    }
    /* Thanh indicator mặc định của Streamlit — ẩn đi, dùng border-bottom tuỳ chỉnh */
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background: transparent !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-border"] {
        background: #21262d !important;
    }

    /* ═══════════════════════════════════════════════
       4. Hero banner
    ═══════════════════════════════════════════════ */
    .hero {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 50%, #58a6ff 100%);
        border-radius: 16px;
        padding: 48px 40px;
        margin-bottom: 32px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(31,111,235,0.35);
    }
    .hero h1 { font-size: 2.6rem; font-weight: 800; color: #fff; margin: 0 0 12px; }
    .hero p  { font-size: 1.1rem; color: rgba(255,255,255,0.88); margin: 0; line-height: 1.7; }

    /* ═══════════════════════════════════════════════
       5. Cards
    ═══════════════════════════════════════════════ */
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        transition: border-color .2s, box-shadow .2s;
    }
    .card:hover { border-color: #388bfd; box-shadow: 0 0 20px rgba(56,139,253,0.18); }
    .card h3 { color: #58a6ff; margin: 0 0 8px; font-size: 1.1rem; }
    .card p  { color: #8b949e; margin: 0; font-size: .93rem; line-height: 1.6; }

    /* ═══════════════════════════════════════════════
       6. Stat pills
    ═══════════════════════════════════════════════ */
    .stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
    .stat-pill {
        flex: 1; min-width: 120px;
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .stat-pill .val { font-size: 2rem; font-weight: 700; color: #58a6ff; line-height: 1; }
    .stat-pill .lbl { font-size: .8rem; color: #8b949e; margin-top: 6px; }

    /* ═══════════════════════════════════════════════
       7. Badge chips
    ═══════════════════════════════════════════════ */
    .chip {
        display: inline-block;
        background: #1f6feb22;
        border: 1px solid #1f6feb;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: .78rem;
        color: #58a6ff;
        margin: 3px;
    }

    /* ═══════════════════════════════════════════════
       8. Detection panel + confidence bars
    ═══════════════════════════════════════════════ */
    .det-panel {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
    }
    .det-header {
        font-size: 1rem; font-weight: 700; color: #58a6ff;
        border-bottom: 1px solid #21262d;
        padding-bottom: 10px; margin-bottom: 14px;
    }
    .conf-bar-wrap { display: flex; align-items: center; gap: 10px; margin: 4px 0; }
    .conf-bar-bg   { flex: 1; background: #21262d; border-radius: 99px; height: 7px; }
    .conf-bar-fill { height: 7px; border-radius: 99px; }

    /* ═══════════════════════════════════════════════
       9. Section title
    ═══════════════════════════════════════════════ */
    .section-title {
        font-size: 1.4rem; font-weight: 700; color: #e6edf3;
        border-left: 4px solid #1f6feb;
        padding-left: 14px; margin: 28px 0 16px;
    }

    /* ═══════════════════════════════════════════════
       10. Misc Streamlit overrides
    ═══════════════════════════════════════════════ */
    div[data-testid="stFileUploader"] {
        background: #161b22 !important;
        border: 2px dashed #30363d !important;
        border-radius: 12px !important;
        transition: border-color .2s;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #388bfd !important;
    }
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* Expander (dùng cho cài đặt nhận diện) */
    [data-testid="stExpander"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────── helpers ─────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str) -> YOLO:
    return YOLO(path)


def confidence_color(conf: float) -> str:
    if conf >= 0.7:
        return "#3fb950"
    elif conf >= 0.45:
        return "#d29922"
    return "#f85149"


CLASS_EMOJI: dict[str, str] = {
    "resistor":   "🟫",
    "capacitor":  "🔷",
    "connector":  "🔌",
    "ic":         "🧠",
    "led":        "💡",
    "diode":      "🔺",
    "transistor": "🔻",
    "inductor":   "🌀",
    "jumper":     "🔗",
    "button":     "🔘",
}

CLASS_DESC: dict[str, str] = {
    "resistor":   "Điện trở — kiểm soát dòng điện",
    "capacitor":  "Tụ điện — lưu trữ năng lượng tạm thời",
    "connector":  "Đầu nối — giao tiếp giữa các khối",
    "ic":         "Vi mạch IC — xử lý & logic",
    "led":        "Đèn LED — tín hiệu quang học",
    "diode":      "Điốt — chỉnh lưu dòng điện",
    "transistor": "Transistor — khuếch đại / chuyển mạch",
    "inductor":   "Cuộn cảm — lọc và lưu trữ từ trường",
    "jumper":     "Cầu nối — kết nối/ngắt mạch",
    "button":     "Nút bấm — tương tác thủ công",
}


# ══════════════════════════════════════════════════════
#  HORIZONTAL TABS — điều hướng chính
# ══════════════════════════════════════════════════════
tab_home, tab_detect, tab_perf, tab_gallery = st.tabs(
    ["🏠  Trang chủ", "🔍  Nhận diện", "📊  Hiệu suất mô hình", "🖼️  Gallery mẫu"]
)


# ══════════════════════════════════════════════════════
#  TAB: HOME
# ══════════════════════════════════════════════════════
with tab_home:
    st.markdown(
        """
        <div class="hero">
            <h1>🔬 PCB Component Detector</h1>
            <p>
                Hệ thống nhận diện linh kiện điện tử trên bo mạch in (PCB) sử dụng<br>
                mô hình <b>YOLOv8</b> huấn luyện trên tập dữ liệu thực tế gồm <b>460 ảnh</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="stat-row">
            <div class="stat-pill"><div class="val">460</div><div class="lbl">Ảnh huấn luyện</div></div>
            <div class="stat-pill"><div class="val">10</div><div class="lbl">Loại linh kiện</div></div>
            <div class="stat-pill"><div class="val">100</div><div class="lbl">Epochs</div></div>
            <div class="stat-pill"><div class="val">640px</div><div class="lbl">Kích thước input</div></div>
            <div class="stat-pill"><div class="val">YOLOv8n</div><div class="lbl">Kiến trúc</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Cách sử dụng</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """<div class="card">
                <h3>① Chọn tab "Nhận diện"</h3>
                <p>Nhấn vào tab <b>Nhận diện</b> trên thanh điều hướng phía trên.</p>
            </div>""", unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """<div class="card">
                <h3>② Tải ảnh PCB lên</h3>
                <p>Kéo thả hoặc chọn ảnh <b>JPG / PNG / BMP / WEBP</b> từ máy tính.</p>
            </div>""", unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """<div class="card">
                <h3>③ Xem kết quả ngay lập tức</h3>
                <p>Mô hình phát hiện và đánh dấu từng linh kiện với <b>tên + độ tin cậy</b>.</p>
            </div>""", unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">10 loại linh kiện được nhận diện</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, (cls, desc) in enumerate(CLASS_DESC.items()):
        with cols[i % 5]:
            emoji = CLASS_EMOJI.get(cls, "🔩")
            st.markdown(
                f"""<div class="card" style="padding:16px; margin-bottom:12px;">
                    <div style="font-size:1.8rem; text-align:center;">{emoji}</div>
                    <h3 style="text-align:center; font-size:.95rem;">{cls}</h3>
                    <p style="text-align:center; font-size:.8rem;">{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Pipeline xử lý</div>', unsafe_allow_html=True)
    pipe_cols = st.columns(4)
    steps = [
        ("📥", "Thu thập dữ liệu", "460 ảnh PCB thực tế, gán nhãn bằng LabelImg"),
        ("🧹", "Tiền xử lý", "Chuẩn hóa bbox, sửa 92 box lỗi, chia 80/10/10%"),
        ("🧠", "Huấn luyện YOLOv8", "Transfer learning từ COCO, Mosaic + Mixup augmentation"),
        ("🔬", "Triển khai", "Streamlit web app, chạy local, inference real-time"),
    ]
    for col, (icon, title, desc) in zip(pipe_cols, steps):
        with col:
            st.markdown(
                f"""<div class="card" style="text-align:center;">
                    <div style="font-size:2rem;">{icon}</div>
                    <h3 style="font-size:.95rem; margin-top:8px;">{title}</h3>
                    <p style="font-size:.82rem;">{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════
#  TAB: DETECT
# ══════════════════════════════════════════════════════
with tab_detect:
    st.markdown('<div class="section-title">Nhận diện linh kiện PCB</div>', unsafe_allow_html=True)

    if not MODEL_PATH.is_file():
        st.error(
            f"Không tìm thấy model tại `{MODEL_PATH}`.\n\n"
            "Hãy đặt file `best.pt` vào thư mục `model/`."
        )
        st.stop()

    # ── Cài đặt nhận diện — inline expander thay cho sidebar ──
    with st.expander("⚙️  Cài đặt nhận diện", expanded=False):
        set_col1, set_col2, set_col3, set_col4 = st.columns([2, 2, 1, 1])
        with set_col1:
            confidence = st.slider(
                "Ngưỡng tin cậy (Conf)",
                min_value=0.05, max_value=0.95, value=0.25, step=0.05,
                help="Chỉ hiển thị dự đoán có độ tin cậy ≥ ngưỡng này.",
            )
        with set_col2:
            iou = st.slider(
                "Ngưỡng chồng lắp (IoU)",
                min_value=0.10, max_value=0.90, value=0.45, step=0.05,
                help="Non-maximum suppression: giảm số box bị trùng.",
            )
        with set_col3:
            show_original = st.toggle("Hiển thị ảnh gốc", value=True)
        with set_col4:
            show_table = st.toggle("Hiển thị bảng chi tiết", value=True)
    # Giá trị mặc định nếu người dùng chưa mở expander
    if "confidence" not in dir():
        confidence, iou, show_original, show_table = 0.25, 0.45, True, True

    uploaded_file = st.file_uploader(
        "Kéo thả hoặc chọn ảnh PCB (JPG · JPEG · PNG · BMP · WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="visible",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div style='text-align:center; padding:60px 0; color:#8b949e;'>
                <div style='font-size:3.5rem;'>📤</div>
                <div style='font-size:1.1rem; margin-top:16px;'>Tải ảnh PCB lên để bắt đầu nhận diện</div>
                <div style='font-size:.85rem; margin-top:8px;'>Hỗ trợ: JPG, JPEG, PNG, BMP, WEBP</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        image = Image.open(uploaded_file).convert("RGB")
        model = load_model(str(MODEL_PATH))

        with st.spinner("Đang phân tích ảnh với YOLOv8…"):
            t0     = time.perf_counter()
            result = model.predict(image, conf=confidence, iou=iou, verbose=False)[0]
            elapsed_ms = (time.perf_counter() - t0) * 1000

        annotated = Image.fromarray(result.plot()[..., ::-1])

        detections: list[dict] = []
        if result.boxes is not None:
            for box in result.boxes:
                cid  = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                detections.append(
                    {
                        "Linh kiện":    result.names[cid],
                        "Tin cậy":      round(conf, 3),
                        "Chiều rộng":   round(w),
                        "Chiều cao":    round(h),
                        "Tọa độ (x1,y1,x2,y2)": f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})",
                    }
                )

        n_det   = len(detections)
        counter = Counter(d["Linh kiện"] for d in detections)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Linh kiện phát hiện", n_det)
        c2.metric("Loại khác nhau", len(counter))
        c3.metric("Thời gian suy luận", f"{elapsed_ms:.0f} ms")
        c4.metric("Ngưỡng tin cậy", f"{confidence:.0%}")

        st.divider()

        img_left, img_right = (st.columns(2) if show_original else (None, st.columns(1)[0]))

        if show_original and img_left:
            with img_left:
                st.markdown("**Ảnh gốc**")
                st.image(image, use_container_width=True)

        with img_right:
            st.markdown("**Kết quả nhận diện**")
            st.image(annotated, use_container_width=True)

        if detections:
            st.divider()
            st.markdown("**Tổng hợp linh kiện phát hiện**")
            chip_html = "".join(
                f'<span class="chip">{CLASS_EMOJI.get(cls, "🔩")} {cls} × {cnt}</span>'
                for cls, cnt in sorted(counter.items(), key=lambda x: -x[1])
            )
            st.markdown(chip_html, unsafe_allow_html=True)

            if show_table:
                st.divider()
                st.markdown("**Chi tiết từng phát hiện**")

                bars_html = '<div class="det-panel"><div class="det-header">Độ tin cậy · Tên linh kiện · Kích thước bounding box</div>'
                for d in sorted(detections, key=lambda x: -x["Tin cậy"]):
                    cv    = d["Tin cậy"]
                    color = confidence_color(cv)
                    emoji = CLASS_EMOJI.get(d["Linh kiện"], "🔩")
                    pct   = int(cv * 100)
                    bars_html += (
                        f'<div class="conf-bar-wrap">'
                        f'  <div style="width:160px; font-size:.85rem;">{emoji} {d["Linh kiện"]}</div>'
                        f'  <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div></div>'
                        f'  <div style="width:42px; font-size:.85rem; color:{color}; text-align:right;">{pct}%</div>'
                        f'  <div style="width:100px; font-size:.78rem; color:#8b949e;">{d["Chiều rộng"]}×{d["Chiều cao"]}px</div>'
                        f'</div>'
                    )
                bars_html += "</div>"
                st.markdown(bars_html, unsafe_allow_html=True)

                st.markdown("")
                df = pd.DataFrame(detections)
                df["Tin cậy"] = df["Tin cậy"].apply(lambda x: f"{x:.1%}")
                st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.warning(
                "Không tìm thấy linh kiện nào với ngưỡng tin cậy hiện tại.\n\n"
                "Thử **giảm ngưỡng tin cậy** trong phần Cài đặt ở trên."
            )

        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        st.download_button(
            label="Tải ảnh kết quả xuống",
            data=buf.getvalue(),
            file_name="pcb_detection_result.png",
            mime="image/png",
        )


# ══════════════════════════════════════════════════════
#  TAB: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
with tab_perf:
    st.markdown('<div class="section-title">Hiệu suất & Đánh giá mô hình</div>', unsafe_allow_html=True)

    st.markdown("### Cấu hình huấn luyện")
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        st.markdown(
            """<div class="card">
                <h3>Kiến trúc</h3>
                <p><b>YOLOv8 Nano (yolov8n)</b><br>
                Pretrained trên COCO (transfer learning)<br>
                Input: 640×640 px</p>
            </div>""", unsafe_allow_html=True
        )
    with cfg2:
        st.markdown(
            """<div class="card">
                <h3>Hyperparameters</h3>
                <p>Epochs: <b>100</b> (patience=20)<br>
                Batch: <b>16</b> · Optimizer: <b>AdamW</b><br>
                LR: <b>0.001</b></p>
            </div>""", unsafe_allow_html=True
        )
    with cfg3:
        st.markdown(
            """<div class="card">
                <h3>Data Augmentation</h3>
                <p>Mosaic=1.0 · Mixup=0.15<br>
                CopyPaste=0.3 · Rotate±90°<br>
                FlipUD=0.5 · FlipLR=0.5</p>
            </div>""", unsafe_allow_html=True
        )

    st.divider()
    st.markdown("### Chỉ số đánh giá theo class")

    if METRICS_CSV.is_file():
        df_m   = pd.read_csv(METRICS_CSV)
        styled = (
            df_m.style
            .format({"Precision": "{:.1%}", "Recall": "{:.1%}", "mAP50": "{:.1%}", "mAP50-95": "{:.1%}"})
            .background_gradient(subset=["Precision", "Recall", "mAP50", "mAP50-95"], cmap="Blues")
            .set_properties(**{"text-align": "center"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        avg = df_m.drop(columns=["Class"]).mean()
        st.markdown("")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Avg Precision",  f"{avg['Precision']:.1%}")
        a2.metric("Avg Recall",     f"{avg['Recall']:.1%}")
        a3.metric("Avg mAP50",      f"{avg['mAP50']:.1%}")
        a4.metric("Avg mAP50-95",   f"{avg['mAP50-95']:.1%}")
    else:
        st.warning("Không tìm thấy file `assest/evaluation_metrics.csv`.")

    st.divider()
    st.markdown("### Đường cong huấn luyện")
    curve_img = GRAPH_DIR / "comprehensive_training_metrics.png"
    if curve_img.is_file():
        st.image(str(curve_img), caption="Box Loss · Cls Loss · Precision · Recall · mAP50 · mAP50-95 theo epoch", use_container_width=True)
    else:
        st.info("Chưa có ảnh `comprehensive_training_metrics.png`.")

    st.divider()
    st.markdown("### Ma trận nhầm lẫn (Confusion Matrix)")
    cm_img = GRAPH_DIR / "confusion_matrix.png"
    if cm_img.is_file():
        st.image(str(cm_img), caption="Confusion Matrix trên tập test", use_container_width=True)
        st.info(
            "Đọc ma trận: Ô chéo chính (màu đậm) = dự đoán đúng. "
            "Cột 'background' = linh kiện không được phát hiện (false negative)."
        )

    st.divider()
    st.markdown("### Thống kê dataset")
    ds1, ds2, ds3, ds4 = st.columns(4)
    ds1.metric("Tổng ảnh hợp lệ", "460")
    ds2.metric("Train set (80%)",  "368 ảnh")
    ds3.metric("Val set (10%)",    "46 ảnh")
    ds4.metric("Test set (10%)",   "46 ảnh")


# ══════════════════════════════════════════════════════
#  TAB: GALLERY
# ══════════════════════════════════════════════════════
with tab_gallery:
    st.markdown('<div class="section-title">Gallery kết quả nhận diện mẫu</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;'>Các ảnh dưới đây là kết quả nhận diện thực tế của mô hình "
        "trên tập test, hiển thị bounding box cùng tên và độ tin cậy của từng linh kiện.</p>",
        unsafe_allow_html=True,
    )

    pred_imgs = sorted(PRED_DIR.glob("prediction_*.jpg"))

    if not pred_imgs:
        st.warning("Chưa có ảnh mẫu trong thư mục `assest/pred/`.")
    else:
        for row_start in range(0, len(pred_imgs), 3):
            cols = st.columns(3)
            for col, img_path in zip(cols, pred_imgs[row_start: row_start + 3]):
                with col:
                    st.image(
                        Image.open(img_path),
                        caption=img_path.stem.replace("_", " ").title(),
                        use_container_width=True,
                    )

    sample_overview = PRED_DIR / "test_predictions_sample.png"
    if sample_overview.is_file():
        st.divider()
        st.markdown("### Tổng hợp ảnh mẫu test")
        st.image(str(sample_overview), caption="test_predictions_sample.png", use_container_width=True)
