import io

import cv2
import numpy as np
from PIL import Image
'''
    HandSpell API image preprocessing module
    This module provides a function to preprocess images for the HandSpell model.
    The preprocessing steps include converting the image to grayscale,
    resizing it to 28x28 pixels, normalizing the pixel values, and reshaping
    it to match the input shape expected by the model.
'''

def preprocess_image(contents: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(contents)) as image:
        grayscale = image.convert("L")
    # Convert the image to a NumPy array for further processing
    pixels = np.array(grayscale)
    # Resize the image to 28x28 pixels
    resized = cv2.resize(pixels, (28, 28))
    # Normalize the pixel values to the range [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Reshape the image to match the input shape expected by the model
    return normalized.reshape(1, 28, 28, 1)
