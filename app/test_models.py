import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_MODEL_PATH = os.path.join(BASE_DIR, 'models/svc_model_TFIDF/model.pkl')
MLB_PATH = os.path.join(BASE_DIR, 'models/mlb_fit.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models/tfidf_vectorizer.pkl')

def load_model(path):
    try:
        with open(path, 'rb') as file:
            model = joblib.load(file)
        print(f"Modèle chargé avec succès : {path}")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle {path}: {str(e)}")
        return None

def main():
    paths = [ML_MODEL_PATH, MLB_PATH, VECTORIZER_PATH]
    
    for path in paths:
        if os.path.exists(path):
            model = load_model(path)
            if model is not None:
                print(f"Type du modèle : {type(model)}")
        else:
            print(f"Le fichier n'existe pas : {path}")

if __name__ == "__main__":
    main()