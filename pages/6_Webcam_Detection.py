"""
SmartVision AI - Real-Time Webcam Detection

Uses the custom-trained YOLOv8s model to detect
objects directly from a webcam video stream.
"""

import av
import cv2
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase,
    RTCConfiguration,
)

from utils.detection import load_detection_model


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Webcam Detection | SmartVision AI",
    layout="wide",
)

st.title("Real-Time Webcam Detection")

st.markdown("""
Use your webcam to detect multiple objects in real time
with the custom-trained YOLOv8s model.

The model can identify objects from the 25 classes
used in this project.
""")


# --------------------------------------------------
# 2. Load YOLO Model
# --------------------------------------------------

@st.cache_resource
def get_yolo_model():
    """Load the trained YOLO model only once."""
    return load_detection_model()


# --------------------------------------------------
# 3. Webcam Video Processor
# --------------------------------------------------

class YOLOVideoProcessor(VideoProcessorBase):

    def __init__(self, model, confidence, iou_threshold):

        self.model = model
        self.confidence = confidence
        self.iou_threshold = iou_threshold

    def recv(self, frame):

        # Convert webcam frame into an OpenCV image
        image = frame.to_ndarray(format="bgr24")

        # Run YOLO object detection
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=640,
            verbose=False
        )

        # Draw bounding boxes and labels
        annotated_image = results[0].plot()

        # Return the processed video frame
        return av.VideoFrame.from_ndarray(
            annotated_image,
            format="bgr24"
        )


# --------------------------------------------------
# 4. Detection Settings
# --------------------------------------------------

st.sidebar.header("Webcam Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05,
)

st.sidebar.info("""
Model: YOLOv8s

Classes: 25

Input: Live Webcam
""")


# --------------------------------------------------
# 5. Webcam Information
# --------------------------------------------------

st.subheader("Live Object Detection")

st.write("""
Click START to activate your webcam. Allow camera permission
when your browser asks for it.
""")

# Load the trained model
try:

    model = get_yolo_model()

    st.success("YOLOv8s model loaded successfully.")

except Exception as error:

    st.error(f"Unable to load YOLO model: {error}")
    st.stop()


# --------------------------------------------------
# 6. WebRTC Configuration
# --------------------------------------------------

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
    }
)


# --------------------------------------------------
# 7. Start Webcam
# --------------------------------------------------

webrtc_streamer(
    key="smartvision-webcam",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=lambda: YOLOVideoProcessor(
        model,
        confidence,
        iou_threshold
    ),
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)


# --------------------------------------------------
# 8. Instructions
# --------------------------------------------------

st.divider()

st.subheader("How to Use")

st.markdown("""
1. Click the **START** button.
2. Allow browser camera permission.
3. Show objects in front of your camera.
4. YOLOv8 will identify objects and draw bounding boxes.
5. Click **STOP** to turn off the webcam.

**Tip:** Use good lighting and keep objects clearly visible
for better detection results.
""")

st.divider()

st.caption(
    "SmartVision AI | Real-Time YOLOv8 Object Detection"
)