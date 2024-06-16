from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import requests

app = FastAPI()

# Function to check if the uploaded image is a leaf using the external API
async def check_leaf(image: bytes):
    url = "https://som11-tomato-leaf-or-not-tomato-leaf.hf.space/predict"
    files = {"fileUploadedByUser": ("image.jpg", image, "image/jpeg")}
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            result = response.json()
            if result.get("predicted_result") == "not_leaf":
                return False
            elif result.get("predicted_result") == "leaf":
                return True
            else:
                return False
        else:
            return False
    except Exception as e:
        print(f"Error checking leaf: {e}")
        return False

# Load the Keras model for tomato disease prediction
tomato_classifier_model = tf.keras.models.load_model('tmtm.h5', compile=False)
tomato_class_names = ["Bacterial-spot", "Early-blight", "Healthy", "Late-blight",
    "Leaf-mold", "Mosaic-virus", "Septoria-leaf-spot", "Yellow-leaf-curl-virus" ]
# Load the Keras model for potato disease prediction
potato_classifier_model = tf.keras.models.load_model('potatoes.h5', compile=False)
potato_class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Function to preprocess image for prediction
def preprocess_image(image):
    image = image.resize((256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Endpoint to test if the server is running
@app.head("/ping")
async def ping():
    return "Hello, I am alive"

# Endpoint to predict tomato disease
@app.post("/predict/tomato")
async def predict_tomato(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Check if the uploaded image is a leaf
    is_leaf = await check_leaf(contents)
    if not is_leaf:
        return JSONResponse(status_code=200, content={"message": "Uploaded image is not a leaf."})

    # Preprocess image
    processed_image = preprocess_image(image)

    # Make prediction using tomato model
    prediction = tomato_classifier_model.predict(processed_image)
    confidence = np.max(prediction) * 100
    predicted_class = tomato_class_names[np.argmax(prediction)]

    return {"class": predicted_class, "confidence": round(confidence, 2)}

# Endpoint to predict potato disease
@app.post("/predict/potato")
async def predict_potato(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Check if the uploaded image is a leaf
    is_leaf = await check_leaf(contents)
    if not is_leaf:
        return JSONResponse(status_code=200, content={"message": "Uploaded image is not a leaf."})

    # Preprocess image
    processed_image = preprocess_image(image)

    # Make prediction using potato model
    prediction = potato_classifier_model.predict(processed_image)
    confidence = np.max(prediction) * 100
    predicted_class = potato_class_names[np.argmax(prediction)]

    return {"class": predicted_class, "confidence": round(confidence, 2)}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Tomato and Potato Disease Prediction API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
