"""
SmartVision AI - Object Detection Page

YOLOv8s-based multi-object detection with:
- Image upload with validation
- Confidence and IoU threshold controls
- Side-by-side original and annotated image
- Detection summary and inference time
- Class-wise statistics and confidence distribution
- Annotated image (PNG) and detection table (CSV) downloads

Requires: streamlit, pandas, pillow, plotly
"""

import time
from collections import Counter
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st
from packaging.version import Version
from PIL import Image, UnidentifiedImageError

from utils.detection import (
    load_detection_model,
    detect_objects,
    extract_detections,
    get_annotated_image,
)


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Object Detection | SmartVision AI",
    layout="wide",
)

# Newer Streamlit versions replace use_container_width with width="stretch"
_ST_VERSION = Version(st.__version__.split("+")[0])
STRETCH = (
    {"width": "stretch"}
    if _ST_VERSION >= Version("1.50.0")
    else {"use_container_width": True}
)

ACCENT = "#1F7A8C"
HIGHLIGHT = "#E09F3E"

MAX_UPLOAD_MB = 15


# --------------------------------------------------
# 2. Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    .stApp, .stMarkdown, h1, h2, h3, h4,
    [data-testid="stMetric"], [data-testid="stCaptionContainer"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1 { font-weight: 600; letter-spacing: -0.02em; }
    h2, h3 { font-weight: 600; letter-spacing: -0.01em; }

    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 8px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] p { font-size: 0.85rem; opacity: 0.75; }
    [data-testid="stMetricValue"] { font-weight: 600; }

    .section-intro { opacity: 0.8; margin-bottom: 1.2rem; max-width: 70ch; }

    .empty-state {
        border: 1px dashed rgba(128, 128, 128, 0.4);
        border-radius: 8px;
        padding: 22px 26px;
        margin-top: 1rem;
    }
    .empty-state p { margin: 0 0 6px 0; }
    .empty-state .lead { font-weight: 600; font-size: 1.05rem; }
    .empty-state .sub  { opacity: 0.8; }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; font-weight: 500; }

    hr { margin: 1.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. Helpers
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading YOLOv8s model...")
def get_yolo_model():
    """Load and cache the trained YOLOv8s model."""
    return load_detection_model()


def image_to_png_bytes(image):
    """Convert a PIL image into PNG bytes for downloading."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def style_fig(fig, height=340):
    """Apply a consistent look to every Plotly figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        title=dict(x=0, xanchor="left", font=dict(size=16)),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


def reset_results():
    """Remove stored results (used when a different image is uploaded)."""
    for key in (
        "detection_results",
        "annotated_image",
        "original_image",
        "inference_ms",
        "run_settings",
    ):
        st.session_state.pop(key, None)


# --------------------------------------------------
# 4. Header
# --------------------------------------------------

st.title("Object Detection")
st.markdown(
    '<p class="section-intro">Upload an image and the custom-trained '
    "YOLOv8s model will locate each object, name its class, and report "
    "a confidence score.</p>",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 5. Sidebar settings
# --------------------------------------------------

st.sidebar.header("Detection settings")

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Minimum confidence required to show a detection. "
    "Higher values show fewer, more certain detections.",
)

iou_threshold = st.sidebar.slider(
    "IoU threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05,
    help="Controls Non-Maximum Suppression. Lower values remove more "
    "overlapping boxes.",
)

st.sidebar.divider()
st.sidebar.caption(
    "Model: YOLOv8s\n\n"
    "Classes: 25 custom classes\n\n"
    "Tip: if nothing is detected, lower the confidence threshold."
)


# --------------------------------------------------
# 6. Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help=f"JPG or PNG, up to {MAX_UPLOAD_MB} MB.",
)

if uploaded_file is None:

    # Clear any results from a previous image
    reset_results()
    st.session_state.pop("file_key", None)

    st.markdown(
        """
        <div class="empty-state">
            <p class="lead">No image uploaded yet</p>
            <p class="sub">Choose a JPG or PNG file above to get started.
            Images with clearly visible objects give the best results.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# Reject oversized files early
file_size_mb = uploaded_file.size / (1024 * 1024)

if file_size_mb > MAX_UPLOAD_MB:
    st.error(
        f"This file is {file_size_mb:.1f} MB. "
        f"Upload an image smaller than {MAX_UPLOAD_MB} MB."
    )
    st.stop()

# Open the image and handle corrupted files
try:
    original_image = Image.open(uploaded_file).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error(
        "The file could not be read as an image. "
        "Upload a valid JPG or PNG file."
    )
    st.stop()

# Clear stored results when a different image is uploaded
file_key = f"{uploaded_file.name}-{uploaded_file.size}"

if st.session_state.get("file_key") != file_key:
    reset_results()
    st.session_state["file_key"] = file_key


# --------------------------------------------------
# 7. Preview and run
# --------------------------------------------------

has_results = (
    "detection_results" in st.session_state
    and st.session_state.get("annotated_image") is not None
)

if not has_results:

    preview_col, info_col = st.columns([2, 1])

    with preview_col:
        st.image(original_image, caption=uploaded_file.name, **STRETCH)

    with info_col:
        st.markdown("#### Image details")
        st.metric("Dimensions", f"{original_image.width} x {original_image.height} px")
        st.metric("File size", f"{file_size_mb:.2f} MB")
        st.metric("Format", uploaded_file.type.split("/")[-1].upper())

run_clicked = st.button(
    "Detect objects" if not has_results else "Run detection again",
    type="primary",
    **STRETCH,
)

if run_clicked:

    try:
        with st.spinner("Detecting objects..."):

            model = get_yolo_model()

            start = time.perf_counter()

            results = detect_objects(
                model=model,
                image=original_image,
                confidence=confidence_threshold,
                iou_threshold=iou_threshold,
            )
            detections = extract_detections(results, model)
            annotated_image = get_annotated_image(results)

            elapsed_ms = (time.perf_counter() - start) * 1000

        st.session_state["detection_results"] = detections
        st.session_state["annotated_image"] = annotated_image
        st.session_state["original_image"] = original_image
        st.session_state["inference_ms"] = elapsed_ms
        st.session_state["run_settings"] = (confidence_threshold, iou_threshold)

        has_results = True

    except Exception as error:
        st.error(
            f"Detection could not be completed: {error}. "
            "Check the model file and try again."
        )


# --------------------------------------------------
# 8. Results
# --------------------------------------------------

if has_results:

    detections = st.session_state["detection_results"]
    annotated_image = st.session_state["annotated_image"]
    run_conf, run_iou = st.session_state["run_settings"]

    st.divider()
    st.subheader("Results")

    # Tell the user if the sliders changed after the last run
    if (run_conf, run_iou) != (confidence_threshold, iou_threshold):
        st.info(
            f"These results used a confidence threshold of {run_conf:.2f} and "
            f"an IoU threshold of {run_iou:.2f}. Select Run detection again "
            "to apply the current settings."
        )

    # ---- Summary metrics ----
    total_objects = len(detections)
    unique_classes = len({item["class_name"] for item in detections})
    average_confidence = (
        sum(item["confidence"] for item in detections) / total_objects
        if total_objects
        else 0
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Objects found", total_objects)
    m2.metric("Unique classes", unique_classes)
    m3.metric("Average confidence", f"{average_confidence * 100:.1f}%")
    m4.metric("Detection time", f"{st.session_state['inference_ms']:.0f} ms")

    # ---- Image comparison ----
    left, right = st.columns(2)

    with left:
        st.image(
            st.session_state["original_image"],
            caption="Original image",
            **STRETCH,
        )

    with right:
        st.image(
            annotated_image,
            caption=f"YOLOv8s detections ({total_objects} found)",
            **STRETCH,
        )

    # ---- Empty detection state ----
    if not detections:
        st.warning(
            "No objects were detected. Lower the confidence threshold in "
            "the sidebar, or try an image with clearer objects."
        )

    # ---- Detailed views ----
    else:
        det_df = pd.DataFrame(
            [
                {
                    "Object": item["class_name"],
                    "Confidence (%)": round(item["confidence"] * 100, 2),
                    "x1": item["x1"],
                    "y1": item["y1"],
                    "x2": item["x2"],
                    "y2": item["y2"],
                }
                for item in detections
            ]
        ).sort_values("Confidence (%)", ascending=False)

        det_df.insert(0, "No.", range(1, len(det_df) + 1))

        tab_classes, tab_table, tab_export = st.tabs(
            ["Class statistics", "All detections", "Download"]
        )

        # ---- Class statistics ----
        with tab_classes:

            class_stats = (
                det_df.groupby("Object")
                .agg(
                    Count=("Object", "size"),
                    Average_confidence=("Confidence (%)", "mean"),
                    Highest_confidence=("Confidence (%)", "max"),
                )
                .reset_index()
                .sort_values("Count", ascending=False)
            )

            c1, c2 = st.columns(2)

            with c1:
                fig = px.bar(
                    class_stats.sort_values("Count"),
                    x="Count",
                    y="Object",
                    orientation="h",
                    text="Count",
                    title="Objects per class",
                )
                fig.update_traces(marker_color=ACCENT, textposition="outside", cliponaxis=False)
                fig.update_xaxes(dtick=1, title_text="")
                fig.update_yaxes(title_text="", showgrid=False)
                st.plotly_chart(style_fig(fig), **STRETCH)

            with c2:
                fig = px.histogram(
                    det_df,
                    x="Confidence (%)",
                    nbins=10,
                    range_x=[0, 100],
                    title="Confidence distribution",
                )
                fig.update_traces(marker_color=HIGHLIGHT)
                fig.update_xaxes(title_text="Confidence (%)")
                fig.update_yaxes(title_text="Detections")
                st.plotly_chart(style_fig(fig), **STRETCH)

            st.dataframe(
                class_stats,
                hide_index=True,
                column_config={
                    "Object": "Class",
                    "Count": st.column_config.NumberColumn("Count", format="%d"),
                    "Average_confidence": st.column_config.ProgressColumn(
                        "Average confidence",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Highest_confidence": st.column_config.ProgressColumn(
                        "Highest confidence",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
                **STRETCH,
            )

        # ---- All detections ----
        with tab_table:

            classes_found = sorted(det_df["Object"].unique())

            selected = st.multiselect(
                "Filter by class",
                options=classes_found,
                default=classes_found,
                help="Filters the table only. The annotated image is unchanged.",
            )

            filtered = det_df[det_df["Object"].isin(selected)]

            st.dataframe(
                filtered,
                hide_index=True,
                column_config={
                    "Confidence (%)": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "x1": st.column_config.NumberColumn("x1", help="Left edge (px)"),
                    "y1": st.column_config.NumberColumn("y1", help="Top edge (px)"),
                    "x2": st.column_config.NumberColumn("x2", help="Right edge (px)"),
                    "y2": st.column_config.NumberColumn("y2", help="Bottom edge (px)"),
                },
                **STRETCH,
            )
            st.caption(
                f"Showing {len(filtered)} of {len(det_df)} detections. "
                "Coordinates are in pixels from the top-left corner."
            )

        # ---- Downloads ----
        with tab_export:

            d1, d2 = st.columns(2)

            with d1:
                st.download_button(
                    label="Download annotated image (PNG)",
                    data=image_to_png_bytes(annotated_image),
                    file_name="smartvision_detection.png",
                    mime="image/png",
                    **STRETCH,
                )

            with d2:
                st.download_button(
                    label="Download detections (CSV)",
                    data=det_df.to_csv(index=False).encode("utf-8"),
                    file_name="smartvision_detections.csv",
                    mime="text/csv",
                    **STRETCH,
                )

    # Annotated image download is still useful with zero detections
    if not detections:
        st.download_button(
            label="Download annotated image (PNG)",
            data=image_to_png_bytes(annotated_image),
            file_name="smartvision_detection.png",
            mime="image/png",
        )


# --------------------------------------------------
# 9. Footer
# --------------------------------------------------

st.divider()
st.caption("SmartVision AI | YOLOv8s Object Detection | 25 custom classes")