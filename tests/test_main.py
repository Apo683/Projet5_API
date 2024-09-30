from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_tags_success():
    # Teste une requête valide pour prédire les tags avec Access Key ID et Secret Access Key
    response = client.post(
        "/predict_tags/",
        json={"Title": "Test title", "Body": "Test body"},
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict_tags_empty_input():
    # Teste une requête avec un texte d'entrée vide
    response = client.post(
        "/predict_tags/",
        json={"Title": "", "Body": ""},
    )
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_tags_route_not_found():
    # Teste une requête vers une route inexistante
    response = client.post("/incorrect_route/", json={"Title": "Test title", "Body": "Test body"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
