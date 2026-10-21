from unittest.mock import MagicMock, patch

import pytest
import torch

from app.services import text_detector


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        text_detector.predict_text("   ")


@pytest.mark.parametrize(
    ("logits", "expected_label"),
    [
        ([0.1, 2.5], "PHISHING"),
        ([2.5, 0.1], "LEGITIMATE"),
    ],
)
def test_prediction_label_and_probabilities(logits, expected_label):
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 2023, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    model = MagicMock()
    model.return_value.logits = torch.tensor([logits])

    with (
        patch.object(
            text_detector,
            "load_model",
            return_value=(tokenizer, model),
        ),
    ):
        result = text_detector.predict_text("Example message")

    assert result["label"] == expected_label
    assert 0.0 <= result["phishing_probability"] <= 1.0
    assert 0.0 <= result["legitimate_probability"] <= 1.0
    assert (
        result["phishing_probability"]
        + result["legitimate_probability"]
    ) == pytest.approx(1.0, abs=0.001)
    assert result["model"] == "distilbert_phishing_classifier"


def test_missing_local_model_raises_clear_error():
    with (
        patch.object(text_detector, "TEXT_MODEL_ID", ""),
        patch.object(text_detector, "LOCAL_MODEL_PATH") as model_path,
    ):
        model_path.exists.return_value = False

        with pytest.raises(RuntimeError, match="Trained DistilBERT model not found"):
            text_detector.load_model()
