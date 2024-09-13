from sentence_transformers import SentenceTransformer

# Charger le modèle sBERT
model_bert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_bert(texts):
    if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Le texte d'entrée pour embed_bert est vide ou invalide.")
    
    # Vectorisation avec sBERT
    return model_bert.encode(texts)