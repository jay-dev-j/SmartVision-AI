# Import required libraries
from PIL import Image

# Import image preprocessing function
from utils.image_processor import preprocess_image

# Import prediction function
from utils.prediction import predict_with_all_models


# Step 1: Create a sample image
sample_image = Image.new(
    "RGB",
    (500, 400),
    color="blue"
)

print("Starting prediction test...")


# Step 2: Preprocess the image
processed_image = preprocess_image(sample_image)

print("Image preprocessing completed!")


# Step 3: Generate predictions using all four models
all_results = predict_with_all_models(processed_image)


# Step 4: Display prediction results
print("\nSMARTVISION AI - CLASSIFICATION RESULTS")

for model_name, results in all_results.items():

    print("\nModel:", model_name)

    print("Predicted class:", results["predicted_class"])

    print(
        "Confidence:",
        f'{results["confidence_percentage"]:.2f}%'
    )

    print("\nTop 5 predictions:")

    for prediction in results["top_predictions"]:

        print(
            f'{prediction["class_name"]}: '
            f'{prediction["confidence_percentage"]:.2f}%'
        )

print("\nAll classification models tested successfully!")
