import joblib
import re
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from app.config import VECTORIZER_PATH

# Charger le vectoriseur TF-IDF (assurez-vous qu'il a été préalablement téléchargé via S3)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

def embed_tfidf(texts):
    if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Le texte d'entrée pour embed_tfidf est vide ou invalide.")
    
    return tfidf_vectorizer.transform(texts)

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

    return text

def remove_stopwords_tokenize_lemmatize(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return tokens