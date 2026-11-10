"""
SmartVision AI - Model Performance Dashboard

Sections:
1. Overview            - best model, ranking, key findings
2. Classification      - metric comparison across four CNN models
3. Training history    - accuracy / loss curves and generalisation gap
4. Object detection    - YOLOv8s metrics, speed benchmark, evaluation plots
5. Dataset             - split sizes and class count

Requires: streamlit, pandas, numpy, plotly
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from packaging.version import Version


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Model Performance | SmartVision AI",
    layout="wide",
)

# Newer Streamlit versions replace use_container_width with width="stretch"
_ST_VERSION = Version(st.__version__.split("+")[0])
STRETCH = (
    {"width": "stretch"}
    if _ST_VERSION >= Version("1.50.0")
    else {"use_container_width": True}
)


# --------------------------------------------------
# 2. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

HISTORY_DIR = BASE_DIR / "assets" / "training_history"
CONFUSION_DIR = BASE_DIR / "assets" / "confusion_matrices"
YOLO_DIR = BASE_DIR / "assets" / "yolo_evaluation"


# --------------------------------------------------
# 3. Design tokens and styling
# --------------------------------------------------

MODEL_COLORS = {
    "VGG16": "#98A2B3",
    "ResNet50": "#1F7A8C",
    "MobileNetV2": "#3D5A80",
    "EfficientNetB0": "#E09F3E",
}

METRIC_COLORS = {
    "Accuracy (%)": "#1F7A8C",
    "Precision (%)": "#3D5A80",
    "Recall (%)": "#E09F3E",
    "F1-Score (%)": "#98A2B3",
}

ACCENT = "#1F7A8C"

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

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 8px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] p { font-size: 0.85rem; opacity: 0.75; }
    [data-testid="stMetricValue"] { font-weight: 600; }

    /* Best-model banner */
    .best-model {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-left: 5px solid #1F7A8C;
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 1rem;
        background: rgba(31, 122, 140, 0.07);
    }
    .best-model .title { font-size: 0.9rem; opacity: 0.75; margin: 0; }
    .best-model .name  { font-size: 1.7rem; font-weight: 600; margin: 2px 0 6px 0; }
    .best-model .desc  { font-size: 0.95rem; margin: 0; opacity: 0.9; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; font-weight: 500; }

    /* Section intro text */
    .section-intro { opacity: 0.8; margin-bottom: 1.2rem; max-width: 70ch; }

    hr { margin: 1.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 4. Helpers
# --------------------------------------------------

def style_fig(fig, height=380):
    """Apply a consistent look to every Plotly figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="IBM Plex Sans, sans-serif", size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            title_text="",
        ),
        title=dict(x=0, xanchor="left", font=dict(size=16)),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


def section_intro(text):
    st.markdown(f'<p class="section-intro">{text}</p>', unsafe_allow_html=True)


def show_image(path, caption):
    """Display an image if it exists, otherwise explain what is missing."""
    if path.exists():
        st.image(str(path), caption=caption, **STRETCH)
    else:
        st.warning(f"{caption} is not available. Expected file: {path.name}")


@st.cache_data(show_spinner=False)
def load_history(path_str):
    """Load a Keras training-history JSON file into a DataFrame."""
    with open(path_str, "r") as file:
        history = json.load(file)

    history_df = pd.DataFrame(history)
    history_df.index = range(1, len(history_df) + 1)
    history_df.index.name = "Epoch"
    return history_df


def history_figure(history_df, train_col, val_col, title, as_percent):
    """Build a training vs validation line chart."""
    scale = 100 if as_percent else 1
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df.index,
            y=history_df[train_col] * scale,
            mode="lines+markers",
            name="Training",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history_df.index,
            y=history_df[val_col] * scale,
            mode="lines+markers",
            name="Validation",
            line=dict(color="#E09F3E", width=2.5, dash="dot"),
            marker=dict(size=5),
        )
    )

    fig.update_layout(title=title)
    fig.update_xaxes(title_text="Epoch", dtick=max(1, len(history_df) // 10))
    fig.update_yaxes(title_text="Accuracy (%)" if as_percent else "Loss")
    return style_fig(fig, height=340)


# --------------------------------------------------
# 5. Data
# --------------------------------------------------

df = pd.DataFrame(
    {
        "Model": ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"],
        "Accuracy (%)": [54.24, 60.69, 56.93, 55.96],
        "Precision (%)": [52.44, 57.53, 57.11, 57.31],
        "Recall (%)": [40.27, 50.28, 46.39, 44.98],
        "F1-Score (%)": [43.02, 51.29, 48.43, 47.39],
    }
)

optimization_df = pd.DataFrame(
    {
        "Format": ["PyTorch", "ONNX", "OpenVINO"],
        "Inference Time (ms)": [684.21, 592.48, 647.78],
        "FPS": [1.462, 1.688, 1.544],
        "Model Size (MB)": [21.49, 42.71, 42.79],
    }
)

TRAINING_HISTORIES = {
    "VGG16": "vgg16_augmented_history.json",
    "ResNet50": "resnet50_finetuned_history.json",
    "MobileNetV2": "mobilenetv2_finetuned_history.json",
}

DATASET_SPLITS = {"Training": 3948, "Validation": 854, "Testing": 931}
NUM_CLASSES = 25

best = df.loc[df["Accuracy (%)"].idxmax()]
fastest = optimization_df.loc[optimization_df["Inference Time (ms)"].idxmin()]


# --------------------------------------------------
# 6. Header
# --------------------------------------------------

st.title("Model Performance")
st.markdown(
    '<p class="section-intro">Evaluation results for the SmartVision AI '
    "25-class recognition system: four classification models and a YOLOv8s "
    "object detector.</p>",
    unsafe_allow_html=True,
)

tab_overview, tab_cls, tab_train, tab_det, tab_data = st.tabs(
    [
        "Overview",
        "Classification",
        "Training history",
        "Object detection",
        "Dataset",
    ]
)


# --------------------------------------------------
# 7. Overview
# --------------------------------------------------

with tab_overview:

    st.markdown(
        f"""
        <div class="best-model">
            <p class="title">Best classification model</p>
            <p class="name">{best['Model']}</p>
            <p class="desc">
                Highest test accuracy ({best['Accuracy (%)']:.2f}%) and
                macro F1-score ({best['F1-Score (%)']:.2f}%) among the
                four models evaluated on the test set.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{best['Accuracy (%)']:.2f}%")
    c2.metric("Precision", f"{best['Precision (%)']:.2f}%")
    c3.metric("Recall", f"{best['Recall (%)']:.2f}%")
    c4.metric("Macro F1-score", f"{best['F1-Score (%)']:.2f}%")

    st.subheader("Model ranking")

    ranking = df.sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
    ranking.insert(0, "Rank", ranking.index + 1)

    pct_col = lambda label: st.column_config.ProgressColumn(
        label, format="%.2f%%", min_value=0, max_value=100
    )

    st.dataframe(
        ranking,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Accuracy (%)": pct_col("Accuracy"),
            "Precision (%)": pct_col("Precision"),
            "Recall (%)": pct_col("Recall"),
            "F1-Score (%)": pct_col("F1-Score"),
        },
        **STRETCH,
    )

    st.subheader("Key findings")

    gap = (df["Precision (%)"] - df["Recall (%)"]).mean()
    spread = df["Accuracy (%)"].max() - df["Accuracy (%)"].min()

    st.markdown(
        f"""
        - **{best['Model']}** leads on accuracy, recall and F1-score. Its lead
          over the lowest-scoring model is {spread:.2f} percentage points of accuracy.
        - Precision is higher than recall for every model, by
          {gap:.1f} points on average. The models are cautious: when they
          predict a class they are usually right, but they miss a share of
          true examples.
        - **YOLOv8s** reaches {37:.0f}% mAP@50 on the cleaned test set, and
          **{fastest['Format']}** gives the fastest CPU inference in the
          export benchmark.
        """
    )


# --------------------------------------------------
# 8. Classification comparison
# --------------------------------------------------

with tab_cls:

    section_intro(
        "Test-set results for the four CNN classifiers. Precision, recall "
        "and F1-score are macro-averaged across the 25 classes."
    )

    left, right = st.columns([1, 1.4])

    with left:
        acc_sorted = df.sort_values("Accuracy (%)")
        fig = go.Figure(
            go.Bar(
                x=acc_sorted["Accuracy (%)"],
                y=acc_sorted["Model"],
                orientation="h",
                marker_color=[MODEL_COLORS[m] for m in acc_sorted["Model"]],
                text=[f"{v:.2f}%" for v in acc_sorted["Accuracy (%)"]],
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig.update_layout(title="Test accuracy")
        fig.update_xaxes(range=[0, 100], title_text="Accuracy (%)")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(style_fig(fig, 380), **STRETCH)

    with right:
        long_df = df.melt(
            id_vars="Model",
            value_vars=["Precision (%)", "Recall (%)", "F1-Score (%)"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(
            long_df,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            text_auto=".1f",
            color_discrete_map=METRIC_COLORS,
            title="Precision, recall and F1-score",
        )
        fig.update_yaxes(range=[0, 75], title_text="Score (%)")
        fig.update_xaxes(title_text="")
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(style_fig(fig, 380), **STRETCH)

    st.subheader("Full comparison")

    st.dataframe(
        df.style.highlight_max(
            subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"],
            color="rgba(31, 122, 140, 0.25)",
        ).format(
            {
                "Accuracy (%)": "{:.2f}",
                "Precision (%)": "{:.2f}",
                "Recall (%)": "{:.2f}",
                "F1-Score (%)": "{:.2f}",
            }
        ),
        hide_index=True,
        **STRETCH,
    )
    st.caption("The best value in each metric column is highlighted.")

    st.subheader("Confusion matrix")

    with st.expander("EfficientNetB0 classification confusion matrix", expanded=False):
        show_image(
            CONFUSION_DIR / "efficientnetb0_confusion_matrix.png",
            "EfficientNetB0 confusion matrix",
        )
        st.caption("Confusion matrices for the other models are not available.")


# --------------------------------------------------
# 9. Training history
# --------------------------------------------------

with tab_train:

    section_intro(
        "Accuracy and loss per epoch on the training and validation sets. "
        "These differ from the final test-set metrics shown elsewhere."
    )

    histories = {}
    for name, filename in TRAINING_HISTORIES.items():
        path = HISTORY_DIR / filename
        if path.exists():
            histories[name] = load_history(str(path))

    model_tabs = st.tabs(list(TRAINING_HISTORIES.keys()) + ["EfficientNetB0"])

    for tab, name in zip(model_tabs[:3], TRAINING_HISTORIES.keys()):
        with tab:

            if name not in histories:
                st.warning(
                    f"Training history file not found for {name}. "
                    f"Expected file: {TRAINING_HISTORIES[name]}"
                )
                continue

            h = histories[name]

            final_train = h["accuracy"].iloc[-1] * 100
            final_val = h["val_accuracy"].iloc[-1] * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Epochs", len(h))
            m2.metric("Final training accuracy", f"{final_train:.2f}%")
            m3.metric("Best validation accuracy", f"{h['val_accuracy'].max() * 100:.2f}%")
            m4.metric(
                "Train-validation gap",
                f"{final_train - final_val:.2f} pts",
                help="Final training accuracy minus final validation accuracy. "
                "A large gap suggests overfitting.",
            )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    history_figure(h, "accuracy", "val_accuracy", "Accuracy", True),
                    **STRETCH,
                )
            with c2:
                st.plotly_chart(
                    history_figure(h, "loss", "val_loss", "Loss", False),
                    **STRETCH,
                )

            with st.expander("View raw epoch data"):
                st.dataframe(h, **STRETCH)

    with model_tabs[3]:
        st.info("Training history is not available for EfficientNetB0.")

    if len(histories) > 1:
        st.subheader("Validation accuracy across models")

        fig = go.Figure()
        for name, h in histories.items():
            fig.add_trace(
                go.Scatter(
                    x=h.index,
                    y=h["val_accuracy"] * 100,
                    mode="lines",
                    name=name,
                    line=dict(color=MODEL_COLORS[name], width=2.5),
                )
            )
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Validation accuracy (%)")
        st.plotly_chart(style_fig(fig, 360), **STRETCH)


# --------------------------------------------------
# 10. Object detection
# --------------------------------------------------

with tab_det:

    section_intro(
        "YOLOv8s trained on the SmartVision AI 25-class dataset and "
        "evaluated on the cleaned test set with PyTorch."
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Precision", "47.50%")
    d2.metric("Recall", "37.16%")
    d3.metric("mAP@50", "36.84%")
    d4.metric("mAP@50-95", "22.15%")

    det_metrics, det_speed, det_plots = st.tabs(
        ["Evaluation plots", "Speed benchmark", "Confidence curves"]
    )

    # ---- Evaluation plots ----
    with det_metrics:

        st.markdown("#### Confusion matrices")
        c1, c2 = st.columns(2)
        with c1:
            show_image(YOLO_DIR / "confusion_matrix.png", "Confusion matrix")
        with c2:
            show_image(
                YOLO_DIR / "confusion_matrix_normalized.png",
                "Normalized confusion matrix",
            )

        st.markdown("#### Precision-recall and F1")
        c1, c2 = st.columns(2)
        with c1:
            show_image(YOLO_DIR / "BoxPR_curve.png", "Precision-recall curve")
        with c2:
            show_image(YOLO_DIR / "BoxF1_curve.png", "F1-score curve")

    # ---- Speed benchmark ----
    with det_speed:

        st.markdown(
            f"""
            Inference speed for three export formats on CPU.
            **{fastest['Format']}** is the fastest at
            {fastest['Inference Time (ms)']:.2f} ms per image
            ({fastest['FPS']:.3f} FPS).
            """
        )

        st.dataframe(
            optimization_df.style.highlight_max(
                subset=["FPS"], color="rgba(31, 122, 140, 0.25)"
            )
            .highlight_min(
                subset=["Inference Time (ms)"], color="rgba(31, 122, 140, 0.25)"
            )
            .format(
                {
                    "Inference Time (ms)": "{:.2f}",
                    "FPS": "{:.3f}",
                    "Model Size (MB)": "{:.2f}",
                }
            ),
            hide_index=True,
            **STRETCH,
        )

        format_colors = {
            "PyTorch": "#98A2B3",
            "ONNX": "#1F7A8C",
            "OpenVINO": "#E09F3E",
        }

        c1, c2, c3 = st.columns(3)

        for col, metric, title in [
            (c1, "Inference Time (ms)", "Inference time (ms)"),
            (c2, "FPS", "Frames per second"),
            (c3, "Model Size (MB)", "Model size (MB)"),
        ]:
            with col:
                fig = px.bar(
                    optimization_df,
                    x="Format",
                    y=metric,
                    color="Format",
                    color_discrete_map=format_colors,
                    text_auto=".2f" if metric != "FPS" else ".3f",
                    title=title,
                )
                fig.update_layout(showlegend=False)
                fig.update_xaxes(title_text="")
                fig.update_yaxes(title_text="")
                fig.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(style_fig(fig, 320), **STRETCH)

        st.caption(
            "ONNX and OpenVINO models are roughly twice the size of the "
            "PyTorch weights in this benchmark."
        )

    # ---- Confidence curves ----
    with det_plots:

        c1, c2 = st.columns(2)
        with c1:
            show_image(YOLO_DIR / "BoxP_curve.png", "Precision vs confidence")
        with c2:
            show_image(YOLO_DIR / "BoxR_curve.png", "Recall vs confidence")


# --------------------------------------------------
# 11. Dataset
# --------------------------------------------------

with tab_data:

    section_intro(
        "The dataset is divided into training, validation and testing "
        "splits across 25 object classes."
    )

    total = sum(DATASET_SPLITS.values())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total images", f"{total:,}")
    k2.metric("Training", f"{DATASET_SPLITS['Training']:,}")
    k3.metric("Validation", f"{DATASET_SPLITS['Validation']:,}")
    k4.metric("Testing", f"{DATASET_SPLITS['Testing']:,}")
    k5.metric("Classes", NUM_CLASSES)

    split_df = pd.DataFrame(
        {
            "Split": list(DATASET_SPLITS.keys()),
            "Images": list(DATASET_SPLITS.values()),
        }
    )
    split_df["Share (%)"] = split_df["Images"] / total * 100

    left, right = st.columns([1, 1])

    with left:
        fig = px.pie(
            split_df,
            names="Split",
            values="Images",
            hole=0.55,
            color="Split",
            color_discrete_map={
                "Training": "#1F7A8C",
                "Validation": "#E09F3E",
                "Testing": "#3D5A80",
            },
            title="Images per split",
        )
        fig.update_traces(textinfo="percent+label", sort=False)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 340), **STRETCH)

    with right:
        st.markdown("#### Split summary")
        st.dataframe(
            split_df,
            hide_index=True,
            column_config={
                "Images": st.column_config.NumberColumn("Images", format="%d"),
                "Share (%)": st.column_config.NumberColumn("Share", format="%.1f%%"),
            },
            **STRETCH,
        )


# --------------------------------------------------
# 12. Footer
# --------------------------------------------------

st.divider()
st.caption(
    "SmartVision AI | Classification and detection model evaluation | "
    f"{NUM_CLASSES} classes"
)