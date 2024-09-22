import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import logging
import gc
import os

# Setting up logging to capture detailed information
log_file_path = "logs/retrain_model.log"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
file_handler = logging.FileHandler(log_file_path)
console_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, console_handler]
)

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Function to free memory
def free_memory():
    gc.collect()
    logging.debug("Memory freed using garbage collector.")

# Function to build the model
def build_model(input_shape_text, input_shape_image, num_classes):
    logging.info("Building model architecture...")
    text_input = Input(shape=(input_shape_text,), name='text_input')
    x1 = Dense(512, activation='relu')(text_input)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(256, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.5)(x1)

    image_input = Input(shape=(input_shape_image,), name='image_input')
    x2 = Dense(256, activation='relu')(image_input)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(128, activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.5)(x2)

    combined = concatenate([x1, x2])
    x = Dense(64, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[text_input, image_input], outputs=output)
    model.compile(optimizer=Nadam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logging.info("Model built successfully.")
    return model

# Function to evaluate the model on test data
def evaluate_model_on_test_data(model, X_test_text, test_image_features, y_test):
    try:
        logging.info("Evaluating model on test data...")
        predictions = model.predict([X_test_text, test_image_features])
        predicted_classes = np.argmax(predictions, axis=1)

        # Calculate F1-score
        f1 = f1_score(y_test, predicted_classes, average='weighted')
        logging.info(f"F1-score on test data: {f1}")

        # Print F1-score to the terminal
        print(f"F1-score on test data: {f1}")

        # Log F1-score to the log file
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"F1-score on test data: {f1}\n")

        return f1
    except Exception as e:
        logging.error(f"Error during F1-score evaluation: {e}")
        return None

# Main function for retraining the model
def retrain_model():
    try:
        # Loading balanced data
        logging.info("Loading balanced data...")
        X_train_text = np.load('src/data/X_train_tfidf_balanced.npy')
        train_image_features = np.load('src/data/train_image_features_balanced.npy')
        y_train = np.load('src/data/Y_train_balanced.npy')

        logging.info("Splitting 10% of the data for the test set...")
        X_train_text, X_test_text, y_train, y_test = train_test_split(X_train_text, y_train, test_size=0.10, random_state=42)
        train_image_features, test_image_features = train_test_split(train_image_features, test_size=0.10, random_state=42)

        free_memory()

        logging.info("Splitting 80% of the remaining training data into train and validation sets...")
        X_train_text, X_val_text, y_train, y_val = train_test_split(X_train_text, y_train, test_size=0.20, random_state=42)
        train_image_features, val_image_features = train_test_split(train_image_features, test_size=0.20, random_state=42)

        logging.debug(f"Training set size: {len(X_train_text)}, Validation set size: {len(X_val_text)}")

        logging.info("Building model...")
        model = build_model(X_train_text.shape[1], train_image_features.shape[1], len(np.unique(y_train)))

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        free_memory()

        logging.info("Starting model training on balanced data...")
        history = model.fit(
            [X_train_text, train_image_features], y_train,
            epochs=30, batch_size=64,
            validation_data=([X_val_text, val_image_features], y_val),
            callbacks=[early_stopping, reduce_lr]
        )

        logging.info("Saving model...")
        model.save('src/models/retrained_balanced_model.keras')
        logging.info("Model saved successfully.")

        # Evaluate the model on the test set
        f1_test = evaluate_model_on_test_data(model, X_test_text, test_image_features, y_test)
        logging.info(f"F1-score on test set: {f1_test}")

    except Exception as e:
        logging.error(f"An error occurred during model retraining: {e}")

if __name__ == "__main__":
    retrain_model()