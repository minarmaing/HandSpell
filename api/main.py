import os

from fastapi import FastAPI, UploadFile
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import cv2
'''
    HandSpell API main module
'''

app = FastAPI()
# Base directory and model path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "kazimh_handspell_mnist_custom_cnn.h5")
print(f"🔄 Attempting to load model from: {os.path.abspath(MODEL_PATH)}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# Prediction endpoint for HandSpell API
@app.post("/predict")
async def predict(file: UploadFile):
    # Read and preprocess the uploaded image
    contents = await file.read()
    # Open the image and convert it to grayscale
    raw_image = Image.open(io.BytesIO(contents)).convert("L")
    # Convert the image to a NumPy array
    # Resize the image to 28x28 pixels
    # Normalize the image to the range [0, 1]
    # Reshape the image to match the model's input shape
    img_np = np.array(raw_image)
    img_resized = cv2.resize(img_np, (28, 28))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_final_tensor = np.reshape(img_normalized, (1, 28, 28, 1))

    # Make predictions using the preprocessed image
    predictions = model.predict(img_final_tensor, verbose=0)
    # Extract the predicted class index and confidence score
    scores = predictions[0]
    # Get the index of the class with the highest score
    predicted_idx = int(np.argmax(scores))
    # Get the confidence score for the predicted class
    confidence = float(scores[predicted_idx])
    # Convert the predicted class index to the corresponding letter
    letter = chr(predicted_idx + ord('A'))
    # Return the prediction result as a JSON response
    return {
        "letter": letter,
        "confidence": confidence
    }
