# Image Classification Using Pre-trained CNN

This project performs image classification to distinguish between cats and dogs using a pre-trained Convolutional Neural Network (CNN) model in TensorFlow/Keras.

## Features

- Utilizes a pre-trained CNN model for image classification.
- Processes images from a specified directory and classifies them as either 'Cat' or 'Dog'.
- Simple and easy-to-use script for batch prediction.

## Requirements

- Python 3.7+
- TensorFlow
- OpenCV
- NumPy

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/image-classification-cnn.git
    cd image-classification-cnn
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install tensorflow opencv-python-headless numpy
    ```

4. **Ensure your pre-trained model is available at the specified path:**

    Place your pre-trained model (`CNNModel.h5`) in the `../Assignment/` directory.

5. **Place your test images in the `../ex/test/` directory.**

## Usage

1. **Run the classification script:**

    ```bash
    python classify_images.py
    ```

2. **Script Output:**

    The script will print the filenames along with their predicted labels (Cat or Dog).

## Code Overview

### Main Script

```python
from tensorflow.keras.models import load_model
import cv2
import os

# Import pre-trained model
model = load_model('../Assignment/CNNModel.h5')

def get_animal(filename):
    A = cv2.imread('../ex/test/' + filename)
    A = cv2.resize(A, (224, 224))
    A = A / 225
    A = A.reshape(1, 224, 224, 3)
    yp = model.predict_on_batch(A).argmax()
    return 'Cat' if yp == 0 else 'Dog'

# Prediction
filename = os.listdir('../ex/test/')
for file in filename:
    print(file, get_animal(file))
