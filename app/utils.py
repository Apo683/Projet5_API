import tensorflow_hub as hub

# Charger le modèle depuis le chemin local
local_model_path = "app/models/universal-sentence-encoder"
model_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
model_use.save(local_model_path)

def embed_use(texts):
    if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Le texte d'entrée pour embed_use est vide ou invalide.")
    return model_use(texts).numpy()