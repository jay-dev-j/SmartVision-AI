# Import required libraries
from PIL import Image

# Import our image preprocessing function
from utils.image_processor import preprocess_image


# Step 1: Create a sample image
# Creating a 500 x 400 blue RGB image for testing
sample_image = Image.new(
    "RGB",
    (500, 400),
    color="blue"
)


print("Starting image preprocessing test...")

# Step 2: Process the sample image
processed_image = preprocess_image(sample_image)


# Step 3: Display the results
print("\nImage preprocessing successful!")

print("Original image size:", sample_image.size)

print("Processed image shape:", processed_image.shape)

print("Processed image data type:", processed_image.dtype)

print("Minimum pixel value:", processed_image.min())

print("Maximum pixel value:", processed_image.max())


# Step 4: Verify the expected output
assert processed_image.shape == (1, 224, 224, 3)

assert processed_image.dtype == "float32"

print("\nAll image preprocessing tests passed!")