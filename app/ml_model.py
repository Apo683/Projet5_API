import joblib
import mlflow
import numpy as np
from app.config import ML_MODEL_PATH, MLB_PATH, MLFLOW_TRACKING_URI

# Charger le modèle et le binarizer une seule fois pour optimiser les performances
mlb = joblib.load(MLB_PATH)

def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(ML_MODEL_PATH)

model = load_model()

def predict_tags(text_embedding, threshold):
    y_prob = model.predict_proba(text_embedding)
    
    if isinstance(y_prob, list):
        y_prob = np.array([prob[:, 1] for prob in y_prob]).T
    
    y_pred_adjusted = (y_prob >= threshold).astype(int)
    
    return mlb.inverse_transform(y_pred_adjusted)