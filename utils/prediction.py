# Import required libraries
import numpy as np

# Import class labels
from utils.class_names import CLASS_NAMES

# Import model loader
from utils.model_loader import load_model


# List of classification models
MODEL_NAMES = [
    "VGG16",
    "ResNet50",
    "MobileNetV2",
    "EfficientNetB0"
]


def predict_image(model, processed_image, top_k=5):
    """
    Generate predictions for a single image.

    Parameters:
        model: Loaded CNN model.
        processed_image: Preprocessed image array.
        top_k: Number of top predictions to return.

    Returns:
        Dictionary containing the top prediction
        and top-k predictions.
    """

    # Generate predictions using the CNN model
    predictions = model.predict(
        processed_image,
        verbose=0
    )

    # Extract probabilities from the first image
    probabilities = predictions[0]

    # Get indices of the top-k predictions
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    # Store the top predictions
    top_predictions = []

    for index in top_indices:

        # Get the class name
        class_name = CLASS_NAMES[index]

        # Get the prediction confidence
        confidence = float(probabilities[index])

        # Add prediction details to the list
        top_predictions.append({
            "class_name": class_name,
            "class_index": int(index),
            "confidence": confidence,
            "confidence_percentage": confidence * 100
        })

    # Get the highest-confidence prediction
    best_prediction = top_predictions[0]

    # Return prediction results
    return {
        "predicted_class": best_prediction["class_name"],
        "confidence": best_prediction["confidence"],
        "confidence_percentage": best_prediction["confidence_percentage"],
        "top_predictions": top_predictions
    }


def predict_with_all_models(processed_image):
    """
    Generate predictions using all four CNN models.

    Returns:
        Dictionary containing results from each model.
    """

    # Store results from all models
    all_results = {}

    # Loop through each classification model
    for model_name in MODEL_NAMES:

        print(f"Running prediction with {model_name}...")

        # Load the selected model
        model = load_model(model_name)

        # Generate predictions
        results = predict_image(
            model,
            processed_image,
            top_k=5
        )

        # Store results
        all_results[model_name] = results

    return all_results