from fastapi.testclient import TestClient
from app.main import app

valid_patient = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "v0.1",
    }


def test_predict():
    with TestClient(app) as client:
        response = client.post("/predict", json=valid_patient)

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_missing_feature():
    invalid_patient = valid_patient.copy()
    del invalid_patient["s6"]

    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_patient)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "s6"]