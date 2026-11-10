# Import required libraries
import numpy as np
from PIL import Image, ImageOps


# Image dimensions used during CNN training
IMAGE_SIZE = (224, 224)


def preprocess_image(uploaded_image):
    """
    Prepare an uploaded image for CNN classification.

    Processing steps:
    1. Open the image.
    2. Convert it to RGB.
    3. Resize while preserving aspect ratio.
    4. Add black padding to make it square.
    5. Convert it into a NumPy array.
    6. Add a batch dimension.

    Returns:
        NumPy array with shape (1, 224, 224, 3).
    """

    # Step 1: Check whether the input is already a PIL Image
    if isinstance(uploaded_image, Image.Image):
        image = uploaded_image
    else:
        image = Image.open(uploaded_image)

    # Step 2: Convert image to RGB format
    image = image.convert("RGB")

    # Step 3: Resize and add black padding
    # This matches the original dataset-generation method
    image = ImageOps.pad(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS,
        color=(0, 0, 0)
    )

    # Step 4: Convert image into a NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Step 5: Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array