from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model, Model
from PIL import Image, ImageOps
import numpy as np
from joblib import load as joblib_load
from tensorflow import expand_dims, convert_to_tensor, float32
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from prometheus_client import Gauge, generate_latest
import sqlite3
import logging

app = FastAPI()

# Initialize logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load text classification model and vectorizer
text_model = load_model('src/models/dnn_classifier.h5')
vectorizer = joblib_load('src/models/Tfidf_Vectorizer.joblib')

# Initialize EfficientNetB0 model for image feature extraction
def build_image_model(input_size=(224, 224, 3)):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_size)
    x = GlobalAveragePooling2D()(base_model.output)  # Apply global pooling to reduce dimensionality
    image_model = Model(inputs=base_model.input, outputs=x)
    return image_model

image_model = build_image_model()

# Preprocessing function for images
def preprocess_image(image, target_size=(224, 224)):
    image = ImageOps.fit(image, target_size, Image.LANCZOS)
    image = np.array(image) / 255.0
    return convert_to_tensor(image, dtype=float32)

# Preprocessing function for text
def preprocess_text_data(designation: str, description: str):
    text = designation + ' ' + description
    return vectorizer.transform([text]).toarray()

# Initialize Prometheus metrics
prediction_accuracy = Gauge('prediction_accuracy', 'Accuracy of model predictions')
prediction_count = Gauge('prediction_count', 'Number of predictions made')

# Initialize SQLite database for logging predictions
conn = sqlite3.connect('predictions.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS predictions (designation TEXT, description TEXT, predicted_class INTEGER)''')
conn.commit()

# API endpoint for predictions
@app.post("/predict/")
async def predict(designation: str = Form(...), description: str = Form(...), file: UploadFile = File(...)):
    # Preprocess text data
    processed_text = preprocess_text_data(designation, description)
    
    # Preprocess image data
    image = Image.open(file.file)
    processed_image = preprocess_image(image)

    # Add batch dimension to the image tensor
    processed_image = expand_dims(processed_image, axis=0)

    # Extract image features using EfficientNetB0
    image_features = image_model.predict(processed_image)

    # Pass both text and image features as inputs to the model
    prediction = text_model.predict([processed_text, image_features])  # Pass as list of inputs
    predicted_class = np.argmax(prediction, axis=1)

    # Log the prediction and input data
    logging.info(f"Designation: {designation}, Description: {description}, Predicted Class: {predicted_class[0]}")
    c.execute("INSERT INTO predictions (designation, description, predicted_class) VALUES (?, ?, ?)", 
              (designation, description, int(predicted_class[0])))
    conn.commit()

    # Update Prometheus metrics
    prediction_count.inc()
    # Placeholder for accuracy update; replace with actual calculation if possible
    prediction_accuracy.set(np.max(prediction))

    # Return the predicted class
    return JSONResponse(content={"predicted_class": int(predicted_class[0])})

# Endpoint for Prometheus to scrape metrics
@app.get("/metrics")
async def get_metrics():
    return JSONResponse(content=generate_latest().decode('utf-8'))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)