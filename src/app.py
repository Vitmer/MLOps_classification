from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from joblib import load as joblib_load
from tensorflow import expand_dims, convert_to_tensor, float32
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

app = FastAPI()

# Corrected paths to model and vectorizer
text_model = load_model('src/models/dnn_classifier.h5')
vectorizer = joblib_load('src/models/Tfidf_Vectorizer.joblib')

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = ImageOps.fit(image, target_size, Image.LANCZOS)
    image = np.array(image) / 255.0
    return convert_to_tensor(image, dtype=float32)

# Preprocess text data
def preprocess_text_data(designation: str, description: str):
    text = designation + ' ' + description
    return vectorizer.transform([text]).toarray()

# Prediction endpoint
@app.post("/predict/")
async def predict(designation: str = Form(...), description: str = Form(...), file: UploadFile = File(...)):
    # Preprocess text
    processed_text = preprocess_text_data(designation, description)
    
    # Preprocess image
    image = Image.open(file.file)
    processed_image = preprocess_image(image)

    # Add batch dimension
    processed_image = expand_dims(processed_image, axis=0)

    # Extract image features
    image_features = image_model.predict(processed_image)

    # Combine text and image features for final prediction
    combined_features = np.concatenate([processed_text, image_features], axis=1)
    prediction = text_model.predict(combined_features)
    predicted_class = np.argmax(prediction, axis=1)

    # Return the prediction
    return JSONResponse(content={"predicted_class": int(predicted_class[0])})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)