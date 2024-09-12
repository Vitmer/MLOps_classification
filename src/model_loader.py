from tensorflow.keras.models import load_model
from joblib import load as joblib_load
from PIL import Image, ImageOps
import numpy as np
from tensorflow import convert_to_tensor, float32
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load text and image models
def load_models():
    text_model = load_model('models/dnn_classifier.h5')
    vectorizer = joblib_load('models/Tfidf_Vectorizer.joblib')
    return text_model, vectorizer

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = ImageOps.fit(image, target_size, Image.LANCZOS)
    image = np.array(image) / 255.0
    return convert_to_tensor(image, dtype=float32)

# Preprocess text
def preprocess_text_data(text, vectorizer):
    return vectorizer.transform([text]).toarray()