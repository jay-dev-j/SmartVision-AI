# --------------------------------------------------
# SmartVision AI - Classification Class Names
# --------------------------------------------------

# IMPORTANT:
# These class names must follow the exact alphabetical
# order assigned by TensorFlow during model training.
#
# Original training code:
# class_names = train_dataset.class_names
#
# TensorFlow image_dataset_from_directory()
# automatically sorts folder names alphabetically.


CLASS_NAMES = [
    "airplane",
    "bed",
    "bench",
    "bicycle",
    "bird",
    "bottle",
    "bowl",
    "bus",
    "cake",
    "car",
    "cat",
    "chair",
    "couch",
    "cow",
    "cup",
    "dog",
    "elephant",
    "horse",
    "motorcycle",
    "person",
    "pizza",
    "potted plant",
    "stop sign",
    "traffic light",
    "truck"
]


# Number of supported classification classes
NUM_CLASSES = len(CLASS_NAMES)


# Display class information
if __name__ == "__main__":

    print("SmartVision AI - Classification Classes")
    print("----------------------------------------")

    print("Total classes:", NUM_CLASSES)

    for index, class_name in enumerate(CLASS_NAMES):

        print(f"{index}: {class_name}")