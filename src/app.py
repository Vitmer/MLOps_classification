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

app = FastAPI()

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

    # Return the predicted class
    return JSONResponse(content={"predicted_class": int(predicted_class[0])})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)