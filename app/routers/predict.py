from fastapi import APIRouter, HTTPException
from app.models import TextRequest
from app.utils import embed_tfidf
from app.utils import clean_text, remove_stopwords_tokenize_lemmatize
from app.ml_model import predict_tags
from app.config import THRESHOLD
import logging

router = APIRouter()

@router.post("/predict_tags/")
async def predict_tags_endpoint(request: TextRequest):
    logging.info(f"Request received: {request}")
    
    # Concaténer le titre et le corps de la question
    full_text = request.Title.strip() + " " + request.Body.strip()
    logging.info(f"Full text: {full_text}")

    # Vérifier si la requête contient un texte vide
    if not full_text.strip():
        raise HTTPException(status_code=422, detail="Le titre et le corps ne peuvent pas être vides.")
    
    # Nettoyage et prétraitement du texte
    cleaned_text = clean_text(full_text)
    tokenized_text = remove_stopwords_tokenize_lemmatize(cleaned_text)
    processed_text = ' '.join(tokenized_text)
    logging.info(f"Processed text: {processed_text}")
    
    # Embedding TF-IDF
    X_predict = embed_tfidf([processed_text])
    predicted_tags = predict_tags(X_predict, THRESHOLD)
    
    return predicted_tags