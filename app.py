from flask import Flask, render_template, request
import os
import pandas as pd
from tensorflow import keras
from PIL import Image
import numpy as np
from keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "./data/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Function to classify skin lesion as malignant or benign
def classify_image(image_path):
    model = keras.Sequential(
        [
            # Define model architecture here
            # Example:
            # keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            # keras.layers.MaxPooling2D((2, 2)),
            # ...
            # keras.layers.Dense(1, activation='sigmoid')  # Binary classification output
        ]
    )

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Load and preprocess the image for classification
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values to the range [0, 1]

    # Perform inference to classify the image
    result = model.predict(img)

    # Check if any element in the result array is greater than or equal to 0.5
    if np.any(result >= 0.5):
        print("result", result)
        classification_result = "Malignant"
    else:
        classification_result = "Benign"

    return classification_result


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the POST request has a file part
        if "file" not in request.files:
            return render_template("index.html", message="No file part")

        file = request.files["file"]

        # If the user submits an empty form
        if file.filename == "":
            return render_template("index.html", message="No selected file")

        # If the file is valid, save it to the uploads directory
        if file:
            filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filename)

            # Perform classification
            result = classify_image(filename)

            # Delete the uploaded file after classification (optional)
            os.remove(filename)

            return render_template("index.html", message=result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
