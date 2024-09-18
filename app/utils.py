import joblib

# Charger le vectoriseur TF-IDF (assurez-vous qu'il a été préalablement téléchargé via S3)
tfidf_vectorizer = joblib.load('/app/models/tfidf_vectorizer.pkl')

def embed_tfidf(texts):
    """
    Vectoriser une liste de textes avec le modèle TF-IDF.
    """
    if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Le texte d'entrée pour embed_tfidf est vide ou invalide.")
    
    # Vectorisation avec TF-IDF
    return tfidf_vectorizer.transform(texts)
