from fastapi import APIRouter
from app.models import TextRequest
from app.utils import embed_tfidf
from app.ml_model import predict_tags
from app.config import THRESHOLD

import logging

router = APIRouter()

@router.post("/predict_tags/")
async def predict_tags_endpoint(request: TextRequest):
    logging.info(f"Request received: {request}")
    
    full_text = request.Title + " " + request.Body
    if not full_text.strip():
        raise ValueError("Le texte d'entrée est vide.")
    
    logging.info(f"Full text: {full_text}")
    
    X_predict = embed_tfidf([full_text])
    predicted_tags = predict_tags(X_predict, THRESHOLD)
    
    return predicted_tags