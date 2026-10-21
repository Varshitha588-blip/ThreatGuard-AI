from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_text_analysis_success():
    mock_result = {
        "label": "PHISHING",
        "phishing_probability": 0.9978,
        "legitimate_probability": 0.0022,
        "model": "distilbert_phishing_classifier",
    }

    with patch("app.main.predict_text", return_value=mock_result):
        response = client.post(
            "/analyze/text",
            json={"text": "Urgent: verify your account immediately."},
        )

    assert response.status_code == 200

    body = response.json()
    assert body["input_type"] == "text"
    assert body["label"] == "PHISHING"
    assert body["phishing_probability"] == 0.9978
    assert "explanation" in body
    assert "warning" in body


def test_empty_text_is_rejected():
    response = client.post(
        "/analyze/text",
        json={"text": ""},
    )

    assert response.status_code == 422


def test_text_model_error_returns_service_unavailable():
    with patch(
        "app.main.predict_text",
        side_effect=RuntimeError("Model unavailable"),
    ):
        response = client.post(
            "/analyze/text",
            json={"text": "Please review this message."},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model unavailable"
