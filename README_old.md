# SmartVision AI

SmartVision AI is a deep learning-based image classification and object detection project developed using a selected 25-class subset of the COCO dataset.

The project uses four CNN architectures for image classification and YOLOv8 for object detection. A Streamlit web application provides an interface to upload images, view predictions, detect objects, and compare model performance.

## Features

- Image classification using four CNN models.
- Top-5 predictions with confidence scores.
- Object detection using YOLOv8.
- Bounding boxes with class labels and confidence scores.
- Adjustable detection confidence and IoU thresholds.
- Classification model performance comparison.
- Training and validation performance visualizations.
- YOLOv8 evaluation visualizations.
- Model optimization comparison.

## Models Used

### Image Classification

1. VGG16
2. ResNet50
3. MobileNetV2
4. EfficientNetB0

### Object Detection

- YOLOv8

## Dataset

The project uses a selected 25-class subset of the COCO dataset.

- Total images: 5,733 cropped object images
- Training images: 3,948
- Validation images: 854
- Testing images: 931
- Number of classes: 25
- Image size: 224 × 224 pixels

The classification dataset was prepared by extracting individual objects using their COCO bounding box annotations.

## Classification Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| VGG16 | 54.24% | 52.44% | 40.27% | 43.02% |
| ResNet50 | 60.69% | 57.53% | 50.28% | 51.29% |
| MobileNetV2 | 56.93% | 57.11% | 46.39% | 48.43% |
| EfficientNetB0 | 55.96% | 57.31% | 44.98% | 47.39% |

ResNet50 achieved the highest test accuracy and macro F1-score among the four classification models.

## YOLOv8 Performance

| Metric | Result |
|---|---:|
| Precision | 47.50% |
| Recall | 37.16% |
| mAP@50 | 36.84% |
| mAP@50-95 | 22.15% |

These results are from the YOLOv8s cleaned test-dataset evaluation.

## Model Optimization

The following formats were compared for CPU inference:

| Format | Average Inference Time | FPS | Model Size |
|---|---:|---:|---:|
| PyTorch | 684.21 ms | 1.462 | 21.49 MB |
| ONNX | 592.48 ms | 1.688 | 42.71 MB |
| OpenVINO | 647.78 ms | 1.544 | 42.79 MB |

ONNX achieved the fastest inference time in this benchmark.

## Technology Stack

- Python
- TensorFlow and Keras
- PyTorch
- Ultralytics YOLOv8
- Streamlit
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Pillow

## Project Structure

```text
streamlit_app/
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── classification/
│   └── detection/
│
├── pages/
│   ├── 2_Image_Classification.py
│   ├── 3_Object_Detection.py
│   └── 4_Model_Performance.py
│
├── utils/
│   ├── class_names.py
│   ├── detection.py
│   ├── image_processor.py
│   ├── model_loader.py
│   └── prediction.py
│
├── assets/
│   ├── confusion_matrices/
│   ├── training_history/
│   └── yolo_evaluation/
│
└── test_images/
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd streamlit_app
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Application Pages

1. **Home:** Project overview and navigation.
2. **Image Classification:** Compare predictions from four CNN models.
3. **Object Detection:** Detect multiple objects using YOLOv8.
4. **Model Performance:** Explore classification and detection evaluation results.

## Project Objective

The main objective of SmartVision AI is to develop an integrated computer vision application that combines image classification and object detection. It also provides a comparison of different deep learning architectures based on their performance.

## Note

Classification and object detection are separate tasks in this application. The classification models predict the main object in an uploaded image, while YOLOv8 detects multiple objects and their locations.

## Author

Developed as part of an AI and Machine Learning project.