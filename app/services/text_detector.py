from pathlib import Path
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_PATH = ROOT / "models" / "text" / "distilbert-phishing"
TEXT_MODEL_ID = os.getenv("TEXT_MODEL_ID", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN") or None

_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    global _tokenizer, _model

    if _model is None or _tokenizer is None:
        if TEXT_MODEL_ID:
            model_source = TEXT_MODEL_ID
            token = HF_TOKEN
        else:
            if not LOCAL_MODEL_PATH.exists():
                raise RuntimeError(
                    "Trained DistilBERT model not found. Place it at "
                    f"{LOCAL_MODEL_PATH} or configure TEXT_MODEL_ID."
                )
            model_source = str(LOCAL_MODEL_PATH)
            token = None

        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                token=token,
            )
            _model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                token=token,
            )
        except Exception as exc:
            _tokenizer = None
            _model = None
            raise RuntimeError(
                f"Could not load the text model from {model_source!r}. "
                "Check the model ID, network access, token permissions, "
                "and available memory."
            ) from exc

        _model.to(_device)
        _model.eval()

    return _tokenizer, _model


def predict_text(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Message text cannot be empty.")

    tokenizer, model = load_model()

    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    inputs = {key: value.to(_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    legitimate_probability = float(probabilities[0].item())
    phishing_probability = float(probabilities[1].item())

    return {
        "label": "PHISHING" if phishing_probability >= 0.5 else "LEGITIMATE",
        "phishing_probability": round(phishing_probability, 4),
        "legitimate_probability": round(legitimate_probability, 4),
        "model": "distilbert_phishing_classifier",
    }
