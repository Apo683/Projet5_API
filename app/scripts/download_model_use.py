import tensorflow_hub as hub

model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
local_model_path = "app/models/universal-sentence-encoder"

# Téléchargement et enregistrement du modèle en local
model_use = hub.load(model_url)
model_use.save(local_model_path)

print(f"Model saved to {local_model_path}")