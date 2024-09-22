from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from tensorflow import expand_dims, convert_to_tensor, float32
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import f1_score
from joblib import load as joblib_load
import subprocess
from prometheus_client import Counter, Summary, REGISTRY
from typing import Optional

# Initialize FastAPI
app = FastAPI()

# OAuth2PasswordBearer route
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Hardcoded users for demonstration
users_db = {
    "user": {"username": "user", "full_name": "Normal User", "email": "user@example.com", "hashed_password": "user", "disabled": False},
    "admin": {"username": "admin", "full_name": "Admin User", "email": "admin@example.com", "hashed_password": "admin", "disabled": False}
}

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

# Function to register or retrieve F1 Score metric from Prometheus
def register_f1_score_metric():
    if 'fastapi_f1_score' not in REGISTRY._names_to_collectors:
        f1_score_summary = Summary('fastapi_f1_score', 'F1 Score of the model')
        logging.info("Metric 'fastapi_f1_score' successfully registered.")
    else:
        f1_score_summary = REGISTRY._names_to_collectors['fastapi_f1_score']
        logging.info("Metric 'fastapi_f1_score' already exists in the Prometheus registry.")
    return f1_score_summary

# Prometheus metrics
prediction_counter = Counter('fastapi_predictions_total', 'Total number of predictions made')

# Register or retrieve F1 Score metric
f1_score_summary = register_f1_score_metric()

# Logging setup
log_file_path = "logs/app.log"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
file_handler = logging.FileHandler(log_file_path)
console_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler, console_handler]
)

logging.info("Logging system initialized successfully.")

# Load the balanced model
model = load_model('src/models/retrained_balanced_model.keras')
logging.info("Retrained balanced model loaded successfully.")

# Load the vectorizer
vectorizer = joblib_load('src/models/Tfidf_vectorizer.joblib')
logging.info("Vectorizer loaded successfully.")

# Load the balanced training data
X_train_text = np.load('src/data/X_train_tfidf_balanced.npy')
train_image_features = np.load('src/data/train_image_features_balanced.npy')
y_train = np.load('src/data/Y_train_balanced.npy')
logging.info(f"Balanced training data loaded successfully. Shape of X_train_text: {X_train_text.shape}, train_image_features: {train_image_features.shape}")

# Function to verify password
def verify_password(plain_password, hashed_password):
    return plain_password == hashed_password

# Function to get user from database
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

# Function to authenticate user
def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# Route to obtain the token
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": form_data.username, "token_type": "bearer"}

# Function to get current user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = get_user(users_db, token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# Admin retrain endpoint
@app.post("/retrain/")
async def retrain_model(token: str = Depends(oauth2_scheme)):
    user = await get_current_user(token)
    if user.username != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")
    
    logging.info("Admin is triggering model retraining.")
    try:
        subprocess.run(["python", "src/retrain_model.py"], check=True)
        return {"status": "Retraining initiated"}
    except subprocess.CalledProcessError as e:
        logging.error(f"Retraining failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Retraining failed")

# Function to preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = ImageOps.fit(image, target_size, Image.LANCZOS)
    image = np.array(image) / 255.0
    image = convert_to_tensor(image, dtype=float32)
    image = preprocess_input(image)
    return expand_dims(image, axis=0)

# Function to calculate F1-score and log it
@f1_score_summary.time()  # Prometheus metric decorator to track time
def calculate_f1_score():
    logging.info("Calculating F1-score...")
    predictions = model.predict([X_train_text, train_image_features])
    predicted_classes = np.argmax(predictions, axis=1)
    
    f1 = f1_score(y_train, predicted_classes, average='weighted')
    logging.info(f"F1-score calculated: {f1}")
    return f1

# Function to save updated training data
def save_updated_training_data(X_train_text_new, train_image_features_new, y_train_new):
    logging.info(f"Saving updated training data... New size will be: {X_train_text_new.shape[0]} samples.")
    np.save('src/data/X_train_tfidf_balanced.npy', X_train_text_new)
    np.save('src/data/train_image_features_balanced.npy', train_image_features_new)
    np.save('src/data/Y_train_balanced.npy', y_train_new)
    logging.info(f"Training data updated and saved successfully.")

# Function to add new data to the training set
def add_new_data_to_training_set(text_features, image_features, predicted_class):
    global X_train_text, train_image_features, y_train

    logging.info(f"Training data size before update: {X_train_text.shape[0]} samples.")

    X_train_text_new = np.vstack([X_train_text, text_features])
    train_image_features_new = np.vstack([train_image_features, image_features])
    y_train_new = np.append(y_train, predicted_class)

    save_updated_training_data(X_train_text_new, train_image_features_new, y_train_new)

    logging.info(f"Training data size after update: {X_train_text_new.shape[0]} samples.")

    X_train_text = X_train_text_new
    train_image_features = train_image_features_new
    y_train = y_train_new

# Background task for updating training data and recalculating F1-score
def background_tasks_after_prediction(processed_text, image_features, predicted_class):
    add_new_data_to_training_set(processed_text, image_features, predicted_class)
    f1_after = calculate_f1_score()
    logging.info(f"New F1-score after update: {f1_after}")
    
    # Automatic retraining if F1-score falls below threshold
    if f1_after < 0.90:
        logging.info("F1-score below threshold. Triggering automatic retraining.")
        try:
            subprocess.run(["python", "src/retrain_model.py"], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Automatic retraining failed: {e}")

# Prediction endpoint
@app.post("/predict/")
async def predict(background_tasks: BackgroundTasks, designation: str = Form(...), description: str = Form(...), file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    user = await get_current_user(token)
    logging.info(f"User {user.username} is making a prediction request.")

    prediction_counter.inc()  # Increment Prometheus counter

    # Preprocess text data
    text_data = designation + ' ' + description
    processed_text = vectorizer.transform([text_data]).toarray()

    # Preprocess image data
    image = Image.open(file.file)
    processed_image = preprocess_image(image)

    # Extract image features using EfficientNetB0
    image_features = EfficientNetB0(weights='imagenet', include_top=False)(processed_image)
    image_features = GlobalAveragePooling2D()(image_features).numpy()

    # Perform prediction
    prediction = model.predict([processed_text, image_features])
    predicted_class = np.argmax(prediction, axis=1)

    logging.info(f"Prediction completed. Predicted class: {predicted_class[0]}")

    # Return the prediction to the client immediately
    response = JSONResponse(content={"predicted_class": int(predicted_class[0])})

    # Background tasks
    background_tasks.add_task(background_tasks_after_prediction, processed_text, image_features, predicted_class[0])

    return response

# Startup event to calculate F1-score
@app.on_event("startup")
async def startup_event():
    current_f1_score = calculate_f1_score()
    logging.info(f"Initial F1-score on startup: {current_f1_score}")