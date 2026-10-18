from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError


LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"

# Resolve relative to this Python file, not the terminal directory.
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "kazimh_cnn_v02.keras"
)

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    if tuple(model.input_shape[1:]) != (28, 28, 1):
        raise ValueError("Expected model input shape (28, 28, 1).")

    if model.output_shape[-1] != len(LETTERS):
        raise ValueError("Expected a model with 24 outputs.")

    app.state.model = model
    yield
    del app.state.model


app = FastAPI(
    title="HandSpell Local CNN",
    lifespan=lifespan,
)


def preprocess_image(contents: bytes) -> np.ndarray:
    with Image.open(BytesIO(contents)) as original:
        oriented = ImageOps.exif_transpose(original)

        # Handle transparent PNGs consistently with training.
        rgba = oriented.convert("RGBA")
        background = Image.new(
            "RGBA",
            rgba.size,
            (255, 255, 255, 255),
        )
        rgb = Image.alpha_composite(
            background, rgba
        ).convert("RGB")

        grayscale = ImageOps.grayscale(rgb)

        prepared = ImageOps.pad(
            grayscale,
            (28, 28),
            method=Image.Resampling.LANCZOS,
            color=255,
        )

        pixels = np.array(prepared, dtype=np.float32) / 255.0

    # Batch, height, width, channels.
    return pixels[np.newaxis, ..., np.newaxis]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "classes": len(LETTERS),
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        contents = file.file.read(MAX_FILE_BYTES + 1)
    finally:
        file.file.close()

    if not contents:
        raise HTTPException(400, "The uploaded file is empty.")

    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(413, "Maximum image size is 10 MB.")

    try:
        inputs = preprocess_image(contents)
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(
            400,
            "Cannot read this image. Upload a valid JPG or PNG.",
        )

    scores = app.state.model(
        inputs,
        training=False,
    ).numpy()[0]

    predicted_index = int(np.argmax(scores))

    return {
        "prediction": LETTERS[predicted_index],
        "confidence": float(scores[predicted_index]),
    }
