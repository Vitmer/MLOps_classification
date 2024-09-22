import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Path to the model
MODEL_PATH = 'src/models/simple_model.h5'

def test_model_prediction():
    # Check if the model file exists
    assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"

    # Load the model
    model = load_model(MODEL_PATH)

    # Create dummy input data (text and image features)
    text_input = np.zeros((1, 5000))  # Assuming 5000 features for text input
    image_input = np.zeros((1, 1280))  # Assuming 1280 features for image input

    # Perform a prediction
    prediction = model.predict([text_input, image_input])

    # Check if the prediction shape is as expected
    expected_shape = (1, model.output_shape[-1])  # Get the number of output classes
    assert prediction.shape == expected_shape, f"Unexpected prediction shape: {prediction.shape}"

    # Check if prediction contains valid probability scores
    assert np.all((prediction >= 0) & (prediction <= 1)), "Prediction values are not valid probabilities."

    # Check if the sum of probabilities is 1
    assert np.isclose(np.sum(prediction), 1.0), "The sum of probabilities is not equal to 1."