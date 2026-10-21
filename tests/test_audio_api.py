from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_audio_upload_genuine_prediction():
    mock_result = {
        "prediction": "genuine",
        "label": 0,
        "sample_rate": 16000,
        "duration_seconds": 4.1,
        "embedding_dimensions": 768,
        "class_probabilities": {
            "genuine": 0.9973,
            "spoof": 0.0027,
        },
    }

    with patch("app.main.predict_audio", return_value=mock_result):
        response = client.post(
            "/analyze/audio",
            files={
                "file": (
                    "sample.wav",
                    BytesIO(b"fake audio bytes"),
                    "audio/wav",
                )
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["input_type"] == "audio"
    assert body["prediction"] == "genuine"
    assert body["label"] == 0
    assert body["embedding_dimensions"] == 768
    assert "warning" in body


def test_audio_upload_spoof_prediction():
    mock_result = {
        "prediction": "spoof",
        "label": 1,
        "sample_rate": 16000,
        "duration_seconds": 7.81,
        "embedding_dimensions": 768,
        "class_probabilities": {
            "genuine": 0.0023,
            "spoof": 0.9977,
        },
    }

    with patch("app.main.predict_audio", return_value=mock_result):
        response = client.post(
            "/analyze/audio",
            files={
                "file": (
                    "sample.wav",
                    BytesIO(b"fake audio bytes"),
                    "audio/wav",
                )
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "spoof"
    assert body["label"] == 1
    assert body["class_probabilities"]["spoof"] == 0.9977


def test_unsupported_audio_extension():
    response = client.post(
        "/analyze/audio",
        files={
            "file": (
                "sample.txt",
                BytesIO(b"not audio"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_empty_audio_upload():
    response = client.post(
        "/analyze/audio",
        files={
            "file": (
                "empty.wav",
                BytesIO(b""),
                "audio/wav",
            )
        },
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_oversized_audio_upload():
    oversized_content = b"x" * (20 * 1024 * 1024 + 1)

    response = client.post(
        "/analyze/audio",
        files={
            "file": (
                "large.wav",
                BytesIO(oversized_content),
                "audio/wav",
            )
        },
    )

    assert response.status_code == 413
    assert "20 MB" in response.json()["detail"]
