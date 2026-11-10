"""
SmartVision AI - YOLO Object Detection

This module handles object detection using the trained YOLOv8s model.
It provides functions for loading the model, detecting objects,
and extracting detection information.
"""

from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import numpy as np


# --------------------------------------------------
# 1. Model Configuration
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "detection"
    / "best.pt"
)


# --------------------------------------------------
# 2. Load YOLO Model
# --------------------------------------------------

def load_detection_model():
    """
    Load the trained YOLOv8s model.

    Returns:
        YOLO: Loaded YOLO model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found at: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    return model


# --------------------------------------------------
# 3. Perform Object Detection
# --------------------------------------------------

def detect_objects(
    model,
    image,
    confidence=0.25,
    iou_threshold=0.45
):
    """
    Detect objects in an uploaded image.

    Args:
        model: Loaded YOLO model.
        image: PIL Image.
        confidence: Minimum confidence threshold.
        iou_threshold: IoU threshold for NMS.

    Returns:
        results: YOLO detection results.
    """

    # Ensure image is in RGB format
    if isinstance(image, Image.Image):
        image = image.convert("RGB")

    else:
        image = Image.open(image).convert("RGB")

    # Convert PIL image to NumPy array
    image_array = np.array(image)

    # Run YOLO object detection
    results = model.predict(
        source=image_array,
        conf=confidence,
        iou=iou_threshold,
        imgsz=640,
        verbose=False
    )

    return results


# --------------------------------------------------
# 4. Extract Detection Information
# --------------------------------------------------

def extract_detections(results, model):
    """
    Extract detected object names, confidence scores,
    and bounding box coordinates.

    Returns:
        list: Detection information.
    """

    detections = []

    for result in results:

        for box in result.boxes:

            # Get class index
            class_id = int(box.cls[0])

            # Get class name
            class_name = model.names[class_id]

            # Get confidence score
            confidence = float(box.conf[0])

            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            })

    return detections


# --------------------------------------------------
# 5. Generate Annotated Image
# --------------------------------------------------

def get_annotated_image(results):
    """
    Generate an image containing bounding boxes,
    class labels, and confidence scores.

    Returns:
        PIL Image: Annotated image.
    """

    for result in results:

        # YOLO returns the annotated image in BGR format
        annotated = result.plot()

        # Convert BGR to RGB
        annotated = annotated[:, :, ::-1]

        return Image.fromarray(annotated)

    return None