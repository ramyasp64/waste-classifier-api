# main.py (FastAPI API version)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import uvicorn
import os

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "waste_classifier.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.txt")
# MODEL_PATH = "model/waste_classifier.h5"
# CLASS_NAMES_PATH = "model/class_names.txt"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize app
app = FastAPI()

# Preprocessing function
def preprocess_image(image_data):
    img = Image.open(BytesIO(image_data)).convert("RGB")
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_tensor = preprocess_image(contents)
        print("Input tensor shape:", input_tensor.shape)
        prediction = model.predict(input_tensor)
        print("Raw prediction output:", prediction)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))
        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
