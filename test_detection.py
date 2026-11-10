"""
SmartVision AI - YOLO Detection Module Test

Tests model loading, object detection,
detection extraction, and annotated image generation.
"""

from utils.detection import (
    load_detection_model,
    detect_objects,
    extract_detections,
    get_annotated_image
)

from pathlib import Path
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load YOLO Model
# --------------------------------------------------

print("Loading YOLOv8s model...")

model = load_detection_model()

print("Model loaded successfully!")
print("Number of classes:", len(model.names))


# --------------------------------------------------
# 2. Load Test Image
# --------------------------------------------------

IMAGE_PATH = Path("test_images/test.jpg")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Test image not found: {IMAGE_PATH}"
    )

print("\nTest image:", IMAGE_PATH)


# --------------------------------------------------
# 3. Perform Object Detection
# --------------------------------------------------

print("\nRunning object detection...")

results = detect_objects(
    model=model,
    image=IMAGE_PATH,
    confidence=0.25,
    iou_threshold=0.45
)

print("Detection completed!")


# --------------------------------------------------
# 4. Extract Detection Information
# --------------------------------------------------

detections = extract_detections(results, model)

print("\nDetected Objects:")
print("-" * 45)

for detection in detections:

    print(
        f"{detection['class_name']}: "
        f"{detection['confidence'] * 100:.2f}%"
    )

print("-" * 45)

print("Total objects detected:", len(detections))


# --------------------------------------------------
# 5. Generate Annotated Image
# --------------------------------------------------

annotated_image = get_annotated_image(results)

if annotated_image is not None:

    print("\nAnnotated image generated successfully!")

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.title("SmartVision AI - YOLOv8s Detection")
    plt.axis("off")
    plt.show()

else:
    print("No annotated image generated.")