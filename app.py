import streamlit as st

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SmartVision AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. Application Header
# --------------------------------------------------

st.title("SmartVision AI")

st.subheader("Intelligent Image Classification and Object Detection")

st.markdown("""
Welcome to **SmartVision AI**, an AI-powered computer vision
application developed using Deep Learning and Computer Vision.

This application combines multiple CNN architectures for image
classification and YOLOv8 for object detection.
""")


# --------------------------------------------------
# 3. Project Overview
# --------------------------------------------------

st.divider()

st.header("Project Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Classification Models",
    "4"
)

col2.metric(
    "Detection Model",
    "YOLOv8"
)

col3.metric(
    "Supported Classes",
    "25"
)

col4.metric(
    "Test Images",
    "931"
)


# --------------------------------------------------
# 4. Key Features
# --------------------------------------------------

st.divider()

st.header("Key Features")

col1, col2, col3 = st.columns(3)

with col1:

    st.subheader("Image Classification")

    st.write("""
    - VGG16
    - ResNet50
    - MobileNetV2
    - EfficientNetB0
    - Top-5 predictions
    """)

with col2:

    st.subheader("Object Detection")

    st.write("""
    - YOLOv8 object detection
    - Bounding box visualization
    - Confidence scores
    - Adjustable detection threshold
    """)

with col3:

    st.subheader("Model Performance")

    st.write("""
    - Accuracy comparison
    - Confusion matrices
    - Inference speed
    - Class-wise performance
    """)

# --------------------------------------------------
# Technology Stack
# --------------------------------------------------

st.divider()

st.header("Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Programming")
    st.markdown("""
    - Python
    - NumPy
    - Pandas
    """)

with col2:
    st.subheader("Deep Learning")
    st.markdown("""
    - TensorFlow
    - Keras
    - PyTorch
    - YOLOv8
    """)

with col3:
    st.subheader("Development")
    st.markdown("""
    - Streamlit
    - OpenCV
    - Matplotlib
    - Google Colab
    """)

# --------------------------------------------------
# 5. Explore SmartVision AI
# --------------------------------------------------

st.divider()

st.header("Explore SmartVision AI")

st.markdown(
    "Choose a feature to get started:"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link(
        "pages/2_Image_Classification.py",
        label="Image Classification",
        use_container_width=True
    )

with col2:
    st.page_link(
        "pages/3_Object_Detection.py",
        label="Object Detection",
        use_container_width=True
    )

with col3:
    st.page_link(
        "pages/4_Model_Performance.py",
        label="Model Performance",
        use_container_width=True
    )

# --------------------------------------------------
# About the Project
# --------------------------------------------------

st.divider()

st.header("About SmartVision AI")

st.markdown("""
SmartVision AI is a computer vision project developed to
explore image classification and object detection using
deep learning techniques.

The project uses four CNN architectures for image
classification and YOLOv8 for detecting multiple objects
within an image.

### Project Objectives

- Compare different CNN architectures for image classification.
- Identify multiple objects using YOLOv8.
- Evaluate classification and detection performance.
- Compare model accuracy, precision, recall and F1-score.
- Explore model optimization for efficient inference.

### Dataset

The project uses a selected 25-class subset of the COCO dataset.

### Classification Models

VGG16, ResNet50, MobileNetV2 and EfficientNetB0.
""")

# --------------------------------------------------
# 5. How to Use
# --------------------------------------------------

st.divider()

st.header("How to Use")

st.markdown("""
1. Open **Image Classification** to classify an uploaded image.
2. Open **Object Detection** to identify multiple objects.
3. Visit **Model Performance** to compare trained models.
4. Explore **About** to learn more about the project.
""")


# --------------------------------------------------
# 6. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "SmartVision AI | Deep Learning & Computer Vision Project"
)