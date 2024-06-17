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

# Load the Keras model for tomato disease prediction using external API
# Mapping of API disease names to desired names
disease_name_mapping = {
    "Tomato___Bacterial_spot": "Bacterial-spot",
    "Tomato___Early_blight": "Early-blight",
    "Tomato___Late_blight": "Late-blight",
    "Tomato___Leaf_Mold": "Leaf-mold",
    "Tomato___Septoria_leaf_spot": "Septoria-leaf-spot",
    "Tomato___Spider_mites": "Spider-mites",
    "Tomato___Target_Spot": "Target-spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow-leaf-curl-virus",
    "Tomato___Tomato_mosaic_virus": "Mosaic-virus",
    "Tomato___healthy": "Healthy"
}

@app.post("/predict/tomato")
async def predict_tomato(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Check if the uploaded image is a leaf
    is_leaf = await check_leaf(contents)
    if not is_leaf:
        return JSONResponse(status_code=200, content={"message": "Uploaded image is not a leaf."})

    url = "https://som11-multimodel-tomato-disease-classification-t-1303b88.hf.space/predict"
    files = {"fileUploadedByUser": ("image.jpg", contents, "image/jpeg")}
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            result = response.json()
            # Extract specific parts of the response
            final_result = result.get("final_predicted_result_of_the_leaf", {})
            predicted_result = final_result.get("predicted_result_returned_by_most_of_the_models")
            confidence_str = final_result.get("maximum_confidence_among_the_common_disease_predicted_by_the_models")
            # Convert confidence from percentage string to float between 0 and 1
            confidence = float(confidence_str.strip('%')) / 100.0

            # Map predicted result to desired name
            predicted_result_mapped = disease_name_mapping.get(predicted_result, predicted_result)
            
            return {
                "class": predicted_result_mapped,
                "confidence": confidence,
            }
        else:
            raise HTTPException(status_code=response.status_code, detail="Prediction failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# Load the Keras model for potato disease prediction (local model)
potato_classifier_model = tf.keras.models.load_model('potatoes.h5', compile=False)
potato_class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Function to preprocess image for prediction
def preprocess_image(image):
    image = image.resize((256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    return image

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
    confidence = float(np.max(prediction[0]))
    predicted_class = potato_class_names[np.argmax(prediction)]

    return {"class": predicted_class, "confidence": confidence}

@app.head("/ping")
async def ping():
    return "Hello, I am alive"

@app.get("/")
async def root():
    return {"message": "Welcome to Tomato and Potato Disease Prediction API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
