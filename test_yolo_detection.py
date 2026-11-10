from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained YOLOv8s model
MODEL_PATH = "models/detection/best.pt"

model = YOLO(MODEL_PATH)

# Change this to the path of your test image
IMAGE_PATH = "test_images/test.jpg"

# Run object detection
results = model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=0.25,
    save=False
)

# Display detection results
for result in results:

    # Print detected objects
    print("\nDetected Objects:")

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(
            f"{class_name}: {confidence * 100:.2f}%"
        )

    # Display annotated image
    annotated_image = result.plot()

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image[:, :, ::-1])
    plt.axis("off")
    plt.title("SmartVision AI - YOLOv8s Detection")
    plt.show()