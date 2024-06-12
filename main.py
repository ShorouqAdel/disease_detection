import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Configure CORS settings
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load the model with compile=False
MODEL_PATH = os.path.join(os.path.dirname(__file__), "potatoes.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    # Open the image using PIL
    image = Image.open(BytesIO(data))
    # Resize the image to expected dimensions (256x256 pixels)
    image = image.resize((256, 256))
    # Convert the image to a numpy array
    image_array = np.array(image)
    return image_array

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # Read and preprocess the uploaded image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # Make predictions using the loaded model
    predictions = MODEL.predict(img_batch)

    # Get the predicted class and confidence score
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    return {
        'class': predicted_class,
        'confidence': confidence
    }

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
