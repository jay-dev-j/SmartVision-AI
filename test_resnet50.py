from pathlib import Path
import tensorflow as tf
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "classification"
    / "resnet50_finetuned_best.keras"
)


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


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

print("Loading ResNet50 model...")

custom_objects = {
    "color_jitter": color_jitter,
    "preprocess_input":
        tf.keras.applications.resnet50.preprocess_input
}

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects,
    compile=False,
    safe_mode=False
)

print("\nResNet50 loaded successfully!")

print("Model name:", model.name)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Total parameters:", model.count_params())

print("\nTesting prediction...")

dummy_image = np.random.randint(
    0,
    256,
    size=(1, 224, 224, 3),
    dtype=np.uint8
)

predictions = model.predict(
    dummy_image,
    verbose=0
)

print("Prediction successful!")
print("Output shape:", predictions.shape)
print("Predicted class index:", np.argmax(predictions[0]))

print("\nResNet50 test completed successfully!")