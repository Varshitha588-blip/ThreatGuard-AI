
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "text" / "distilbert-phishing"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    messages = [
        "Urgent! Verify your password immediately to avoid account suspension.",
        "Hi, the project review has been moved to Friday at 10 AM.",
    ]

    for message in messages:
        inputs = tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probabilities).item())
        predicted_label = model.config.id2label[predicted_id]

        print("\nMessage:", message)
        print("Prediction:", predicted_label)
        print("Class probabilities:")

        for class_id, probability in enumerate(probabilities.tolist()):
            label = model.config.id2label[class_id]
            print(f"  {label}: {probability:.4f}")


if __name__ == "__main__":
    main()
