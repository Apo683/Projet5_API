import mlflow
import joblib
import numpy as np
import boto3
from app.config import MLB_PATH, MLFLOW_TRACKING_URI, ML_MODEL_PATH

# Initialiser l'URI de suivi MLflow (Tracking URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Charger le binarizer une seule fois pour optimiser les performances
mlb = joblib.load(MLB_PATH)

def load_model():
    return mlflow.sklearn.load_model(ML_MODEL_PATH)

# Charger le modèle
model = load_model()

def predict_tags(text_embedding, threshold):
    y_prob = model.predict_proba(text_embedding)
    
    # Si la sortie est une liste, la transformer en un tableau numpy
    if isinstance(y_prob, list):
        y_prob = np.array([prob[:, 1] for prob in y_prob]).T
    
    # Ajuster la prédiction en fonction du seuil
    y_pred_adjusted = (y_prob >= threshold).astype(int)
    
    # Retourner les prédictions inversées (tags)
    return mlb.inverse_transform(y_pred_adjusted)