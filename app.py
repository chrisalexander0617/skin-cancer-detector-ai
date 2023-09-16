from flask import Flask, render_template, request
import os
import pandas as pd
from tensorflow import keras
from PIL import Image
import numpy as np

import deeplake  # Assuming you have a deep learning model for classification

app = Flask(__name__)

# Specify the directory where uploaded images will be stored
UPLOAD_FOLDER = './data/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to classify skin lesion as malignant or benign
def classify_image(image_path):
    # Load your deep learning model here
    model = create_model()  # Create your model using create_model function

    # Load model weights from a checkpoint (if applicable)
    # model.load_weights('model_weights.h5')

    # Load and preprocess the image for classification
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image to match the model's input size
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values to the range [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Perform inference to classify the image
    result = model.predict(img)

    # Assuming a binary classification, you can return 'Malignant' or 'Benign' based on the result
    if result[0][0] >= 0.5:
        classification_result = 'Malignant'
    else:
        classification_result = 'Benign'

    return classification_result

def create_model():
    model = keras.Sequential([
        # Define your model architecture here
        # Example:
        # keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        # keras.layers.MaxPooling2D((2, 2)),
        # ...
        # keras.layers.Dense(1, activation='sigmoid')  # Binary classification output
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    # Check if the POST request has a file part
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    # If the user submits an empty form
    if file.filename == '':
        return render_template('index.html', message='No selected file')

    # If the file is valid, save it to the uploads directory
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Perform classification
        result = classify_image(filename)

        # Delete the uploaded file after classification (optional)
        os.remove(filename)

        return render_template('index.html', message=result)

  return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
