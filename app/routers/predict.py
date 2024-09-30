from fastapi import APIRouter, HTTPException
from app.models import TextRequest
from app.utils import embed_tfidf
from app.ml_model import predict_tags
from app.config import THRESHOLD

import logging

router = APIRouter()

@router.post("/predict_tags/")
async def predict_tags_endpoint(request: TextRequest):
    logging.info(f"Request received: {request}")
    
    # Concaténer le titre et le corps de la question
    full_text = request.Title.strip() + " " + request.Body.strip()

    # Vérifier si la requête contient un texte vide
    if not full_text.strip():
        raise HTTPException(status_code=422, detail="Le titre et le corps ne peuvent pas être vides.")
    
    logging.info(f"Full text: {full_text}")
    
    X_predict = embed_tfidf([full_text])
    predicted_tags = predict_tags(X_predict, THRESHOLD)
    
    return predicted_tags