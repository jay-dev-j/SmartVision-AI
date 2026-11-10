from pathlib import Path
import tensorflow as tf

# Register custom color jitter function
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


# Model path
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "classification"
    / "vgg16_augmented_best.keras"
)

print("Loading VGG16 model...")

# Load trained model
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "color_jitter": color_jitter,
        "preprocess_input": tf.keras.applications.vgg16.preprocess_input
    },
    compile=False
)

print("VGG16 model loaded successfully!")
model.summary()

print("\nModel Information:")
print("Input Shape:", model.input_shape)
print("Output Shape:", model.output_shape)
print("Total Parameters:", model.count_params())