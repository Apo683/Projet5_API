import joblib
import numpy as np
from app.config import MLB_PATH, ML_MODEL_PATH

# Charger le binarizer une seule fois pour optimiser les performances
mlb = joblib.load(MLB_PATH)

def load_model():
    with open(ML_MODEL_PATH, 'r', encoding='utf-8') as f:
        return joblib.load(ML_MODEL_PATH)

# Charger le modèle
model = load_model()

def predict_tags(text_embedding, threshold):
    # Convertir les données en format dense si nécessaire
    text_embedding_dense = text_embedding.toarray() if hasattr(text_embedding, 'toarray') else text_embedding
    y_prob = model.predict_proba(text_embedding_dense)
        
    # Si la sortie est une liste, la transformer en un tableau numpy
    if isinstance(y_prob, list):
        y_prob = np.array([prob[:, 1] for prob in y_prob]).T
    
    # Ajuster la prédiction en fonction du seuil
    y_pred_adjusted = (y_prob >= threshold).astype(int)
    
    # Retourner les prédictions inversées (tags)
    return mlb.inverse_transform(y_pred_adjusted)