"""
PCB Component Detection — Streamlit Web Application
Light theme (forced) · Horizontal tabs · Schematic SVG icons
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
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════
#  CSS — Force light theme + custom styling
# ══════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ── Force light mode at every level ── */
    :root { color-scheme: light only !important; }

    html, body {
        background: #ffffff !important;
        color: #1f2328 !important;
    }

    /* Streamlit root containers */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    [data-testid="stMain"],
    .main {
        background: #ffffff !important;
        color: #1f2328 !important;
    }

    /* Header bar ở trên cùng (nguồn gốc gây viền màu lạ) */
    header[data-testid="stHeader"] {
        background: #ffffff !important;
        border-bottom: 1px solid #d0d7de !important;
        box-shadow: none !important;
    }
    [data-testid="stToolbar"] { background: #ffffff !important; }

    /* Ẩn sidebar + hamburger */
    [data-testid="stSidebar"]          { display: none !important; }
    #MainMenu                          { visibility: hidden !important; }
    [data-testid="collapsedControl"]   { display: none !important; }
    button[title="View fullscreen"]    { display: none !important; }

    /* Block container — padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] {
        background: #ffffff;
        border-bottom: 1px solid #d0d7de;
        margin-bottom: 28px;
    }
    button[data-baseweb="tab"] {
        background: transparent !important;
        color: #57606a !important;
        font-size: .93rem !important;
        font-weight: 500 !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 10px 22px !important;
        border: none !important;
        transition: color .15s, background .15s !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #0969da !important;
        background: #f6f8fa !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0969da !important;
        background: #ffffff !important;
        border-bottom: 2px solid #0969da !important;
        font-weight: 700 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: transparent !important; }
    [data-testid="stTabs"] [data-baseweb="tab-border"]    { background: #d0d7de !important; }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #0550ae 0%, #0969da 55%, #388bfd 100%);
        border-radius: 16px;
        padding: 52px 40px;
        margin-bottom: 32px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(9,105,218,0.22);
    }
    .hero h1 { font-size: 2.6rem; font-weight: 800; color: #fff; margin: 0 0 12px; }
    .hero p  { font-size: 1.1rem; color: rgba(255,255,255,.9); margin: 0; line-height: 1.8; }

    /* ── Stat pills ── */
    .stat-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 28px; }
    .stat-pill {
        flex: 1; min-width: 120px;
        background: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
    }
    .stat-pill .val { font-size: 1.8rem; font-weight: 700; color: #0969da; line-height: 1; }
    .stat-pill .lbl { font-size: .8rem; color: #57606a; margin-top: 6px; }

    /* ── Cards ── */
    .card {
        background: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 14px;
        transition: border-color .18s, box-shadow .18s;
    }
    .card:hover { border-color: #0969da; box-shadow: 0 0 0 3px rgba(9,105,218,0.10); }
    .card h3 { color: #0969da; margin: 0 0 8px; font-size: 1rem; font-weight: 700; }
    .card p  { color: #57606a; margin: 0; font-size: .9rem; line-height: 1.6; }

    /* ── Component card (icon + name) ── */
    .comp-card {
        background: #ffffff;
        border: 1px solid #d0d7de;
        border-radius: 12px;
        padding: 20px 12px 16px;
        margin-bottom: 12px;
        text-align: center;
        transition: border-color .18s, box-shadow .18s;
    }
    .comp-card:hover { border-color: #0969da; box-shadow: 0 0 0 3px rgba(9,105,218,0.10); }
    .comp-card .comp-name { font-size: .92rem; font-weight: 700; color: #1f2328; margin: 10px 0 4px; }
    .comp-card .comp-desc { font-size: .78rem; color: #57606a; }

    /* ── Badge chips ── */
    .chip {
        display: inline-block;
        background: #ddf4ff;
        border: 1px solid #54aeff;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: .8rem;
        color: #0550ae;
        font-weight: 600;
        margin: 3px;
    }

    /* ── Section title ── */
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #1f2328;
        border-left: 4px solid #0969da;
        padding-left: 14px; margin: 28px 0 18px;
    }

    /* ── Detection panel ── */
    .det-panel {
        background: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 12px;
        padding: 20px;
    }
    .det-header {
        font-size: .95rem; font-weight: 700; color: #0969da;
        border-bottom: 1px solid #d0d7de;
        padding-bottom: 10px; margin-bottom: 14px;
    }
    .conf-bar-wrap { display: flex; align-items: center; gap: 10px; margin: 5px 0; }
    .conf-bar-bg   { flex: 1; background: #d0d7de; border-radius: 99px; height: 7px; }
    .conf-bar-fill { height: 7px; border-radius: 99px; }

    /* ── Streamlit widgets in light mode ── */
    div[data-testid="stFileUploader"] {
        background: #f6f8fa !important;
        border: 2px dashed #d0d7de !important;
        border-radius: 12px !important;
        transition: border-color .2s;
    }
    div[data-testid="stFileUploader"]:hover { border-color: #0969da !important; }

    [data-testid="stExpander"] {
        background: #f6f8fa !important;
        border: 1px solid #d0d7de !important;
        border-radius: 10px !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 10px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] > div { color: #57606a !important; font-size: .82rem; }
    [data-testid="stMetricValue"] > div { color: #0969da !important; font-weight: 700; }

    /* Dataframe */
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #d0d7de; }

    /* Info / warning / error boxes */
    [data-testid="stAlert"] { border-radius: 10px !important; }

    /* Divider */
    hr { border-color: #d0d7de !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════
#  SVG Schematic Icons — ký hiệu điện tử chuẩn
# ══════════════════════════════════════════════════════
_C = "#0969da"   # accent blue

COMPONENT_SVG: dict[str, str] = {
    # Resistor — ký hiệu ANSI zigzag
    "resistor": f"""
        <svg viewBox="0 0 90 36" xmlns="http://www.w3.org/2000/svg" width="80" height="32">
          <line x1="0" y1="18" x2="14" y2="18" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <polyline points="14,18 19,6 25,30 31,6 37,30 43,6 49,30 55,6 61,30 67,18 76,18"
            fill="none" stroke="{_C}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
          <line x1="76" y1="18" x2="90" y2="18" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
        </svg>""",

    # Capacitor — hai thanh song song
    "capacitor": f"""
        <svg viewBox="0 0 50 60" xmlns="http://www.w3.org/2000/svg" width="42" height="52">
          <line x1="25" y1="2" x2="25" y2="24" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="5" y1="24" x2="45" y2="24" stroke="{_C}" stroke-width="4"   stroke-linecap="round"/>
          <line x1="5" y1="36" x2="45" y2="36" stroke="{_C}" stroke-width="4"   stroke-linecap="round"/>
          <line x1="25" y1="36" x2="25" y2="58" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
        </svg>""",

    # Connector — ổ cắm 2 chân
    "connector": f"""
        <svg viewBox="0 0 60 54" xmlns="http://www.w3.org/2000/svg" width="56" height="50">
          <rect x="4" y="6" width="28" height="42" rx="4"
            fill="none" stroke="{_C}" stroke-width="2.5"/>
          <circle cx="18" cy="20" r="5" fill="{_C}"/>
          <circle cx="18" cy="36" r="5" fill="{_C}"/>
          <line x1="32" y1="20" x2="56" y2="20" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="32" y1="36" x2="56" y2="36" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
        </svg>""",

    # IC — hình chữ nhật với chân ra hai bên
    "ic": f"""
        <svg viewBox="0 0 60 56" xmlns="http://www.w3.org/2000/svg" width="56" height="52">
          <rect x="14" y="8" width="32" height="40" rx="3"
            fill="#ddf4ff" stroke="{_C}" stroke-width="2.5"/>
          <line x1="14" y1="18" x2="4"  y2="18" stroke="{_C}" stroke-width="2" stroke-linecap="round"/>
          <line x1="14" y1="28" x2="4"  y2="28" stroke="{_C}" stroke-width="2" stroke-linecap="round"/>
          <line x1="14" y1="38" x2="4"  y2="38" stroke="{_C}" stroke-width="2" stroke-linecap="round"/>
          <line x1="46" y1="18" x2="56" y2="18" stroke="{_C}" stroke-width="2" stroke-linecap="round"/>
          <line x1="46" y1="28" x2="56" y2="28" stroke="{_C}" stroke-width="2" stroke-linecap="round"/>
          <line x1="46" y1="38" x2="56" y2="38" stroke="{_C}" stroke-width="2" stroke-linecap="round"/>
          <text x="30" y="32" text-anchor="middle" font-size="9" font-weight="800"
            font-family="monospace" fill="{_C}">IC</text>
        </svg>""",

    # LED — diode + tia sáng
    "led": f"""
        <svg viewBox="0 0 72 50" xmlns="http://www.w3.org/2000/svg" width="66" height="46">
          <line x1="0"  y1="25" x2="14" y2="25" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <polygon points="14,10 14,40 40,25" fill="{_C}"/>
          <line x1="40" y1="10" x2="40" y2="40" stroke="{_C}" stroke-width="3"   stroke-linecap="round"/>
          <line x1="40" y1="25" x2="58" y2="25" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="46" y1="8"  x2="54" y2="2"  stroke="{_C}" stroke-width="2"   stroke-linecap="round"/>
          <polyline points="54,2 54,6 50,6" fill="none" stroke="{_C}" stroke-width="1.6" stroke-linecap="round"/>
          <line x1="52" y1="16" x2="60" y2="10" stroke="{_C}" stroke-width="2"   stroke-linecap="round"/>
          <polyline points="60,10 60,14 56,14" fill="none" stroke="{_C}" stroke-width="1.6" stroke-linecap="round"/>
        </svg>""",

    # Diode — tam giác + thanh chặn
    "diode": f"""
        <svg viewBox="0 0 70 42" xmlns="http://www.w3.org/2000/svg" width="64" height="38">
          <line x1="0"  y1="21" x2="16" y2="21" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <polygon points="16,8 16,34 42,21" fill="{_C}"/>
          <line x1="42" y1="8"  x2="42" y2="34" stroke="{_C}" stroke-width="3"   stroke-linecap="round"/>
          <line x1="42" y1="21" x2="70" y2="21" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
        </svg>""",

    # Transistor — NPN trong vòng tròn
    "transistor": f"""
        <svg viewBox="0 0 62 62" xmlns="http://www.w3.org/2000/svg" width="58" height="58">
          <circle cx="36" cy="31" r="22" fill="none" stroke="{_C}" stroke-width="2.5"/>
          <line x1="2"  y1="31" x2="24" y2="31" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="24" y1="16" x2="24" y2="46" stroke="{_C}" stroke-width="3"   stroke-linecap="round"/>
          <line x1="24" y1="22" x2="52" y2="10" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="24" y1="40" x2="52" y2="52" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <polygon points="46,9 52,10 48,16" fill="{_C}"/>
        </svg>""",

    # Inductor — cuộn xoắn
    "inductor": f"""
        <svg viewBox="0 0 90 36" xmlns="http://www.w3.org/2000/svg" width="84" height="32">
          <line x1="0"  y1="18" x2="10" y2="18" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <path d="M10,18 C14,18 14,6 19,6 C24,6 24,18 28,18
                   C32,18 32,6 37,6 C42,6 42,18 46,18
                   C50,18 50,6 55,6 C60,6 60,18 64,18
                   C68,18 68,6 73,6 C78,6 78,18 82,18"
            fill="none" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="82" y1="18" x2="90" y2="18" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
        </svg>""",

    # Jumper — cầu nối (short circuit bridge)
    "jumper": f"""
        <svg viewBox="0 0 70 50" xmlns="http://www.w3.org/2000/svg" width="64" height="46">
          <line x1="0"  y1="36" x2="14" y2="36" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <circle cx="18" cy="36" r="4" fill="none" stroke="{_C}" stroke-width="2.5"/>
          <line x1="22" y1="36" x2="48" y2="36" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <circle cx="52" cy="36" r="4" fill="none" stroke="{_C}" stroke-width="2.5"/>
          <line x1="56" y1="36" x2="70" y2="36" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <path d="M18,32 Q35,10 52,32" fill="none" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
        </svg>""",

    # Button — nút nhấn thường hở (NO push button)
    "button": f"""
        <svg viewBox="0 0 70 52" xmlns="http://www.w3.org/2000/svg" width="64" height="48">
          <line x1="0"  y1="38" x2="18" y2="38" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="18" y1="38" x2="18" y2="24" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="12" y1="24" x2="28" y2="24" stroke="{_C}" stroke-width="3"   stroke-linecap="round"/>
          <line x1="52" y1="38" x2="70" y2="38" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="52" y1="38" x2="52" y2="24" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="42" y1="24" x2="58" y2="24" stroke="{_C}" stroke-width="3"   stroke-linecap="round"/>
          <line x1="30" y1="38" x2="52" y2="38" stroke="{_C}" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="35" y1="15" x2="35" y2="6"  stroke="{_C}" stroke-width="2"   stroke-linecap="round"/>
          <line x1="30" y1="6"  x2="40" y2="6"  stroke="{_C}" stroke-width="2"   stroke-linecap="round"/>
        </svg>""",
}

CLASS_DESC: dict[str, str] = {
    "resistor":   "Điện trở",
    "capacitor":  "Tụ điện",
    "connector":  "Đầu nối",
    "ic":         "Vi mạch IC",
    "led":        "Đèn LED",
    "diode":      "Điốt",
    "transistor": "Transistor",
    "inductor":   "Cuộn cảm",
    "jumper":     "Cầu nối",
    "button":     "Nút bấm",
}

CLASS_SUBDESC: dict[str, str] = {
    "resistor":   "Kiểm soát dòng điện",
    "capacitor":  "Lưu trữ năng lượng",
    "connector":  "Giao tiếp giữa các khối",
    "ic":         "Xử lý & logic",
    "led":        "Tín hiệu quang học",
    "diode":      "Chỉnh lưu dòng điện",
    "transistor": "Khuếch đại / chuyển mạch",
    "inductor":   "Lọc và lưu trữ từ trường",
    "jumper":     "Kết nối / ngắt mạch",
    "button":     "Tương tác thủ công",
}


# ─────────────────────── helpers ─────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str) -> YOLO:
    return YOLO(path)


def confidence_color(conf: float) -> str:
    if conf >= 0.7:
        return "#1a7f37"   # green
    elif conf >= 0.45:
        return "#9a6700"   # amber
    return "#cf222e"       # red


# ══════════════════════════════════════════════════════
#  TABS
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
                Hệ thống nhận diện linh kiện điện tử trên bo mạch in (PCB)<br>
                sử dụng mô hình <b>YOLOv8</b> huấn luyện trên <b>460 ảnh</b> thực tế.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stat pills
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

    # How to use
    st.markdown('<div class="section-title">Cách sử dụng</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    steps_home = [
        ("① Chọn tab Nhận diện", 'Nhấn tab <b>Nhận diện</b> trên thanh điều hướng phía trên.'),
        ("② Tải ảnh PCB lên",    'Kéo thả hoặc chọn ảnh <b>JPG / PNG / BMP / WEBP</b>.'),
        ("③ Xem kết quả",        'Mô hình đánh dấu linh kiện với <b>tên + độ tin cậy</b>.'),
    ]
    for col, (title, body) in zip([c1, c2, c3], steps_home):
        with col:
            st.markdown(
                f'<div class="card"><h3>{title}</h3><p>{body}</p></div>',
                unsafe_allow_html=True,
            )

    # Component catalog — SVG schematic icons
    st.markdown('<div class="section-title">10 loại linh kiện được nhận diện</div>', unsafe_allow_html=True)
    cols5 = st.columns(5)
    for i, cls in enumerate(CLASS_DESC):
        with cols5[i % 5]:
            svg  = COMPONENT_SVG.get(cls, "")
            name = CLASS_DESC[cls]
            sub  = CLASS_SUBDESC[cls]
            st.markdown(
                f"""<div class="comp-card">
                    <div style="display:flex;justify-content:center;align-items:center;height:52px;">
                        {svg}
                    </div>
                    <div class="comp-name">{name}</div>
                    <div class="comp-desc">{sub}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # Pipeline
    st.markdown('<div class="section-title">Pipeline xử lý</div>', unsafe_allow_html=True)
    pipe_cols = st.columns(4)
    pipe_steps = [
        ("📥", "Thu thập dữ liệu",   "460 ảnh PCB thực tế, gán nhãn bằng LabelImg"),
        ("🧹", "Tiền xử lý",         "Chuẩn hóa bbox, sửa 92 box lỗi, chia 80/10/10%"),
        ("🧠", "Huấn luyện YOLOv8",  "Transfer learning từ COCO, Mosaic + Mixup augmentation"),
        ("🔬", "Triển khai",          "Streamlit web app, chạy local, inference real-time"),
    ]
    for col, (icon, title, desc) in zip(pipe_cols, pipe_steps):
        with col:
            st.markdown(
                f"""<div class="card" style="text-align:center;">
                    <div style="font-size:2rem;">{icon}</div>
                    <h3 style="font-size:.93rem;margin-top:8px;">{title}</h3>
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

    # Settings — inline expander
    with st.expander("⚙️  Cài đặt nhận diện", expanded=False):
        sc1, sc2, sc3, sc4 = st.columns([2, 2, 1, 1])
        with sc1:
            confidence = st.slider("Ngưỡng tin cậy (Conf)",
                0.05, 0.95, 0.25, 0.05,
                help="Chỉ hiển thị box có độ tin cậy ≥ ngưỡng.")
        with sc2:
            iou = st.slider("Ngưỡng IoU (NMS)",
                0.10, 0.90, 0.45, 0.05,
                help="Non-maximum suppression — giảm box trùng lặp.")
        with sc3:
            show_original = st.toggle("Hiển thị ảnh gốc", value=True)
        with sc4:
            show_table = st.toggle("Bảng chi tiết", value=True)

    uploaded_file = st.file_uploader(
        "Kéo thả hoặc chọn ảnh PCB (JPG · JPEG · PNG · BMP · WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.markdown(
            """<div style='text-align:center; padding:60px 0; color:#57606a;'>
                <div style='font-size:3rem;'>📤</div>
                <div style='font-size:1.1rem; margin-top:14px; font-weight:600;'>
                    Tải ảnh PCB lên để bắt đầu nhận diện
                </div>
                <div style='font-size:.85rem; margin-top:6px;'>
                    Hỗ trợ: JPG, JPEG, PNG, BMP, WEBP
                </div>
            </div>""",
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
                detections.append({
                    "Linh kiện":           result.names[cid],
                    "Tin cậy":             round(conf, 3),
                    "Chiều rộng":          round(w),
                    "Chiều cao":           round(h),
                    "Tọa độ (x1,y1,x2,y2)": f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})",
                })

        counter = Counter(d["Linh kiện"] for d in detections)

        # Quick metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Linh kiện phát hiện", len(detections))
        m2.metric("Loại khác nhau",      len(counter))
        m3.metric("Thời gian suy luận",  f"{elapsed_ms:.0f} ms")
        m4.metric("Ngưỡng tin cậy",      f"{confidence:.0%}")

        st.divider()

        # Images
        if show_original:
            left, right = st.columns(2)
            with left:
                st.markdown("**Ảnh gốc**")
                st.image(image, use_container_width=True)
            with right:
                st.markdown("**Kết quả nhận diện**")
                st.image(annotated, use_container_width=True)
        else:
            st.markdown("**Kết quả nhận diện**")
            st.image(annotated, use_container_width=True)

        if detections:
            st.divider()
            st.markdown("**Tổng hợp linh kiện phát hiện**")
            chips = "".join(
                f'<span class="chip">{CLASS_DESC.get(cls, cls)} × {cnt}</span>'
                for cls, cnt in sorted(counter.items(), key=lambda x: -x[1])
            )
            st.markdown(chips, unsafe_allow_html=True)

            if show_table:
                st.divider()
                st.markdown("**Chi tiết từng phát hiện**")
                bars = '<div class="det-panel"><div class="det-header">Độ tin cậy · Tên linh kiện · Kích thước bounding box</div>'
                for d in sorted(detections, key=lambda x: -x["Tin cậy"]):
                    cv    = d["Tin cậy"]
                    color = confidence_color(cv)
                    pct   = int(cv * 100)
                    vn    = CLASS_DESC.get(d["Linh kiện"], d["Linh kiện"])
                    bars += (
                        f'<div class="conf-bar-wrap">'
                        f'  <div style="width:150px;font-size:.85rem;color:#1f2328;">{vn}</div>'
                        f'  <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div></div>'
                        f'  <div style="width:40px;font-size:.85rem;color:{color};text-align:right;font-weight:700;">{pct}%</div>'
                        f'  <div style="width:100px;font-size:.78rem;color:#57606a;">{d["Chiều rộng"]}×{d["Chiều cao"]}px</div>'
                        f'</div>'
                    )
                bars += "</div>"
                st.markdown(bars, unsafe_allow_html=True)

                st.markdown("")
                df = pd.DataFrame(detections)
                df["Tin cậy"] = df["Tin cậy"].apply(lambda x: f"{x:.1%}")
                st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.warning(
                "Không tìm thấy linh kiện nào với ngưỡng tin cậy hiện tại.\n\n"
                "Thử **giảm Ngưỡng tin cậy** trong phần Cài đặt ở trên."
            )

        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        st.download_button(
            label="⬇️  Tải ảnh kết quả xuống",
            data=buf.getvalue(),
            file_name="pcb_detection_result.png",
            mime="image/png",
        )


# ══════════════════════════════════════════════════════
#  TAB: PERFORMANCE
# ══════════════════════════════════════════════════════
with tab_perf:
    st.markdown('<div class="section-title">Hiệu suất & Đánh giá mô hình</div>', unsafe_allow_html=True)

    st.markdown("### Cấu hình huấn luyện")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""<div class="card">
            <h3>Kiến trúc</h3>
            <p><b>YOLOv8 Nano (yolov8n)</b><br>
            Pretrained trên COCO (transfer learning)<br>Input: 640×640 px</p>
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown("""<div class="card">
            <h3>Hyperparameters</h3>
            <p>Epochs: <b>100</b> (patience=20)<br>
            Batch: <b>16</b> · Optimizer: <b>AdamW</b><br>LR: <b>0.001</b></p>
        </div>""", unsafe_allow_html=True)
    with p3:
        st.markdown("""<div class="card">
            <h3>Data Augmentation</h3>
            <p>Mosaic=1.0 · Mixup=0.15<br>
            CopyPaste=0.3 · Rotate±90°<br>FlipUD=0.5 · FlipLR=0.5</p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Chỉ số đánh giá theo class")

    if METRICS_CSV.is_file():
        df_m = pd.read_csv(METRICS_CSV)
        styled = (
            df_m.style
            .format({"Precision": "{:.1%}", "Recall": "{:.1%}", "mAP50": "{:.1%}", "mAP50-95": "{:.1%}"})
            .background_gradient(subset=["Precision", "Recall", "mAP50", "mAP50-95"], cmap="Blues")
            .set_properties(**{"text-align": "center"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        avg = df_m.drop(columns=["Class"]).mean()
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Avg Precision",  f"{avg['Precision']:.1%}")
        a2.metric("Avg Recall",     f"{avg['Recall']:.1%}")
        a3.metric("Avg mAP50",      f"{avg['mAP50']:.1%}")
        a4.metric("Avg mAP50-95",   f"{avg['mAP50-95']:.1%}")
    else:
        st.warning("Không tìm thấy file `assest/evaluation_metrics.csv`.")

    st.divider()
    st.markdown("### Đường cong huấn luyện")
    curve = GRAPH_DIR / "comprehensive_training_metrics.png"
    if curve.is_file():
        st.image(str(curve), caption="Box Loss · Cls Loss · Precision · Recall · mAP50 · mAP50-95 theo epoch", use_container_width=True)
    else:
        st.info("Chưa có ảnh `comprehensive_training_metrics.png`.")

    st.divider()
    st.markdown("### Ma trận nhầm lẫn (Confusion Matrix)")
    cm = GRAPH_DIR / "confusion_matrix.png"
    if cm.is_file():
        st.image(str(cm), caption="Confusion Matrix trên tập test", use_container_width=True)
        st.info("Ô chéo chính (màu đậm) = dự đoán đúng. Cột 'background' = linh kiện bị bỏ sót (false negative).")

    st.divider()
    st.markdown("### Thống kê dataset")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Tổng ảnh hợp lệ", "460")
    d2.metric("Train (80%)",  "368 ảnh")
    d3.metric("Val (10%)",    "46 ảnh")
    d4.metric("Test (10%)",   "46 ảnh")


# ══════════════════════════════════════════════════════
#  TAB: GALLERY
# ══════════════════════════════════════════════════════
with tab_gallery:
    st.markdown('<div class="section-title">Gallery kết quả nhận diện mẫu</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#57606a;'>Kết quả nhận diện thực tế trên tập test — "
        "bounding box kèm tên linh kiện và độ tin cậy.</p>",
        unsafe_allow_html=True,
    )

    pred_imgs = sorted(PRED_DIR.glob("prediction_*.jpg"))
    if not pred_imgs:
        st.warning("Chưa có ảnh mẫu trong `assest/pred/`.")
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

    sample = PRED_DIR / "test_predictions_sample.png"
    if sample.is_file():
        st.divider()
        st.markdown("### Tổng hợp ảnh mẫu test")
        st.image(str(sample), caption="test_predictions_sample.png", use_container_width=True)
