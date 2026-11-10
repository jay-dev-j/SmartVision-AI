import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="About - SmartVision AI",
    layout="wide"
)

# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("About SmartVision AI")

st.markdown(
    """
    **SmartVision AI** is a deep learning-based computer vision
    application that combines image classification and object detection.

    The application uses four CNN architectures for image classification
    and YOLOv8 for detecting multiple objects in images.
    """
)

st.divider()

# --------------------------------------------------
# PROJECT OVERVIEW
# --------------------------------------------------

st.header("Project Overview")

st.write(
    """
    The main objective of SmartVision AI is to develop an integrated
    computer vision system that can identify individual objects and
    detect multiple objects in real-world images.

    The project uses a selected 25-class subset of the COCO dataset.
    Transfer learning is applied to train and compare different CNN
    architectures, while YOLOv8 is used for object detection.
    """
)

# --------------------------------------------------
# PROJECT STATISTICS
# --------------------------------------------------

st.header("Project Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Object Classes", "25")

with col2:
    st.metric("Classification Models", "4")

with col3:
    st.metric("Test Images", "931")

with col4:
    st.metric("Detection Model", "YOLOv8")

st.divider()

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.header("Dataset Information")

st.write(
    """
    We used a selected subset of the **COCO (Common Objects in Context)**
    dataset.

    The dataset contains everyday objects from different categories.
    COCO bounding box annotations were used to extract individual
    objects for the classification task.
    """
)

dataset_col1, dataset_col2 = st.columns(2)

with dataset_col1:
    st.subheader("Classification Dataset")

    st.markdown(
        """
        - Total images: **5,733**
        - Training images: **3,948**
        - Validation images: **854**
        - Testing images: **931**
        - Number of classes: **25**
        - Input image size: **224 × 224**
        """
    )

with dataset_col2:
    st.subheader("Dataset Split")

    st.markdown(
        """
        - Training: **70%**
        - Validation: **15%**
        - Testing: **15%**

        Individual objects were cropped using bounding box
        annotations and organised into class-specific folders.
        """
    )

with st.expander("View all 25 object classes"):

    class_names = [
        "Airplane", "Bed", "Bench", "Bicycle", "Bird",
        "Bottle", "Bowl", "Bus", "Cake", "Car",
        "Cat", "Chair", "Couch", "Cow", "Cup",
        "Dog", "Elephant", "Horse", "Motorcycle",
        "Person", "Pizza", "Potted Plant", "Stop Sign",
        "Traffic Light", "Truck"
    ]

    for i in range(0, len(class_names), 5):
        st.write(", ".join(class_names[i:i + 5]))

st.divider()

# --------------------------------------------------
# MODEL ARCHITECTURES
# --------------------------------------------------

st.header("Deep Learning Models")

st.subheader("Image Classification")

st.write(
    """
    Four pretrained CNN architectures were implemented using
    transfer learning and adapted for our 25-class classification task.
    """
)

models = {
    "VGG16": "A convolutional neural network using multiple 3 × 3 filters.",
    "ResNet50": "A residual network that uses skip connections to support deep learning.",
    "MobileNetV2": "An efficient CNN designed for mobile and resource-constrained devices.",
    "EfficientNetB0": "A CNN architecture that uses compound scaling for efficiency."
}

for model_name, description in models.items():
    with st.expander(model_name):
        st.write(description)

st.subheader("Object Detection")

st.write(
    """
    **YOLOv8s** is used to detect multiple objects in an image.
    It predicts object classes, bounding box coordinates and
    confidence scores.
    """
)

# --------------------------------------------------
# CLASSIFICATION PERFORMANCE
# --------------------------------------------------

st.divider()

st.header("Classification Model Performance")

st.write(
    "The following table shows the classification results "
    "obtained on the test dataset."
)

performance_data = {
    "Model": [
        "VGG16",
        "ResNet50",
        "MobileNetV2",
        "EfficientNetB0"
    ],
    "Accuracy": [
        "54.24%",
        "60.69%",
        "56.93%",
        "55.96%"
    ],
    "Precision": [
        "52.44%",
        "57.53%",
        "57.11%",
        "57.31%"
    ],
    "Recall": [
        "40.27%",
        "50.28%",
        "46.39%",
        "44.98%"
    ],
    "F1-Score": [
        "43.02%",
        "51.29%",
        "48.43%",
        "47.39%"
    ]
}

st.dataframe(
    performance_data,
    use_container_width=True,
    hide_index=True
)

st.success(
    "ResNet50 achieved the highest test accuracy of 60.69% "
    "and the highest macro F1-score of 51.29%."
)

# --------------------------------------------------
# YOLO PERFORMANCE
# --------------------------------------------------

st.header("YOLOv8 Detection Performance")

yolo_col1, yolo_col2, yolo_col3, yolo_col4 = st.columns(4)

with yolo_col1:
    st.metric("Precision", "47.50%")

with yolo_col2:
    st.metric("Recall", "37.16%")

with yolo_col3:
    st.metric("mAP@0.5", "36.84%")

with yolo_col4:
    st.metric("mAP@0.5:0.95", "22.15%")

st.caption(
    "Results from the YOLOv8s cleaned test-dataset evaluation."
)

# --------------------------------------------------
# TECHNOLOGY STACK
# --------------------------------------------------

st.divider()

st.header("Technology Stack")

tech_col1, tech_col2 = st.columns(2)

with tech_col1:
    st.subheader("Programming and Frameworks")

    st.markdown(
        """
        - Python
        - TensorFlow
        - Keras
        - PyTorch
        - Ultralytics YOLOv8
        """
    )

with tech_col2:
    st.subheader("Application and Tools")

    st.markdown(
        """
        - Streamlit
        - NumPy
        - Pandas
        - OpenCV
        - Pillow
        - Matplotlib
        - Google Colab
        """
    )

# --------------------------------------------------
# PROJECT OBJECTIVE
# --------------------------------------------------

st.divider()

st.header("Project Objective")

st.write(
    """
    SmartVision AI demonstrates how deep learning can be used
    for image classification and object detection.

    It also provides a comparison of different CNN architectures
    and explores model inference optimisation using different
    deployment formats.
    """
)

# --------------------------------------------------
# DEVELOPER INFORMATION
# --------------------------------------------------

st.divider()

st.header("Developer Information")

st.markdown(
    """
    **Developed by:** Jayadev J.

    **Project:** SmartVision AI

    **Domain:** Artificial Intelligence, Machine Learning
    and Computer Vision

    **Academic Background:** B.Tech in Computer Science and Engineering
    """
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SmartVision AI | Deep Learning-Based Image Classification "
    "and Object Detection"
)