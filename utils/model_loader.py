from pathlib import Path
import tensorflow as tf
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models" / "classification"


@tf.keras.utils.register_keras_serializable()
def color_jitter(images):

    images = images / 255.0

    images = tf.image.random_hue(
        images,
        max_delta=0.05
    )

    images = tf.image.random_saturation(
        images,
        lower=0.8,
        upper=1.2
    )

    images = tf.clip_by_value(
        images,
        0.0,
        1.0
    )

    images = images * 255.0

    return images


MODEL_FILES = {
    "VGG16": "vgg16_augmented_best.keras",
    "ResNet50": "resnet50_finetuned_best.keras",
    "MobileNetV2": "mobilenetv2_finetuned_best.keras",
    "EfficientNetB0": "efficientnetb0_best.keras"
}


CUSTOM_OBJECTS = {
    "color_jitter": color_jitter,
    "preprocess_input": tf.keras.applications.vgg16.preprocess_input
}


@st.cache_resource
def load_model(model_name):

    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    custom_objects = CUSTOM_OBJECTS.copy()

    if model_name == "ResNet50":
        custom_objects["preprocess_input"] = (
            tf.keras.applications.resnet50.preprocess_input
        )

    elif model_name == "MobileNetV2":
        custom_objects["preprocess_input"] = (
            tf.keras.applications.mobilenet_v2.preprocess_input
        )

    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )

    return model


def load_all_models():

    models = {}

    for model_name in MODEL_FILES:
        models[model_name] = load_model(model_name)

    return models