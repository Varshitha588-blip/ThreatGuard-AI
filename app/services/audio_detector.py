from pathlib import Path
import os

import joblib
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = os.getenv(
    "AUDIO_MODEL_ID",
    "facebook/wav2vec2-base",
).strip() or "facebook/wav2vec2-base"

HF_TOKEN = os.getenv("HF_TOKEN") or None

CLASSIFIER_PATH = (
    ROOT / "models" / "audio" / "cross_generator_wav2vec2_classifier.joblib"
)

_device = torch.device("cpu")
_extractor = None
_encoder = None
_classifier = None


def _load_models():
    global _extractor, _encoder, _classifier

    if (
        _extractor is not None
        and _encoder is not None
        and _classifier is not None
    ):
        return

    if not CLASSIFIER_PATH.exists():
        raise RuntimeError(
            f"Audio classifier not found: {CLASSIFIER_PATH}"
        )

    try:
        _extractor = AutoFeatureExtractor.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
        )
        _encoder = AutoModel.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
        )
        _encoder.to(_device)
        _encoder.eval()
        _classifier = joblib.load(CLASSIFIER_PATH)

    except Exception:
        _extractor = None
        _encoder = None
        _classifier = None
        raise


def predict_audio(audio_path: str) -> dict:
    path = Path(audio_path)

    if not path.is_file():
        raise ValueError("The supplied audio file does not exist.")

    try:
        _load_models()

        audio, sample_rate = librosa.load(
            path,
            sr=16000,
            mono=True,
        )

        if audio.size == 0:
            raise ValueError("The audio file is empty.")

        if not np.isfinite(audio).all():
            raise ValueError("The audio contains invalid values.")

        inputs = _extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(_device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = _encoder(**inputs)

        embedding = (
            outputs.last_hidden_state.mean(dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if embedding.shape != (768,):
            raise RuntimeError(
                f"Unexpected embedding shape: {embedding.shape}"
            )

        embedding_batch = embedding.reshape(1, -1)
        prediction = int(_classifier.predict(embedding_batch)[0])

        probabilities = None
        if hasattr(_classifier, "predict_proba"):
            probabilities = _classifier.predict_proba(embedding_batch)[0]

        result = {
            "prediction": "spoof" if prediction == 1 else "genuine",
            "label": prediction,
            "sample_rate": 16000,
            "duration_seconds": round(len(audio) / 16000, 2),
            "embedding_dimensions": int(embedding.shape[0]),
        }

        if probabilities is not None:
            classes = list(_classifier.classes_)
            result["class_probabilities"] = {
                "genuine": round(
                    float(probabilities[classes.index(0)]), 4
                ),
                "spoof": round(
                    float(probabilities[classes.index(1)]), 4
                ),
            }

        return result

    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Audio analysis failed: {type(exc).__name__}: {exc}"
        ) from exc
