from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_tags():
    response = client.post("/predict_tags/", json={"Title": "Test title", "Body": "Test body"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)