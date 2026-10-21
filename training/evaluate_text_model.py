from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "text" / "distilbert-phishing"
TEST_FILE = ROOT / "data" / "processed" / "transformer" / "test.csv"
OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("THREATGUARD AI - TEXT MODEL EVALUATION")
print("=" * 65)

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Saved model not found: {MODEL_DIR}")

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_FILE}")

df = pd.read_csv(TEST_FILE).dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Expected binary labels 0 and 1.")

print(f"Test samples: {len(df)}")
print("True label counts:")
print(df["label"].value_counts().sort_index().to_string())

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Device: {device}")
print("Running predictions...")

texts = df["text"].tolist()
true_labels = df["label"].to_numpy()
all_predictions = []
all_probabilities = []
batch_size = 16
max_length = 256

with torch.inference_mode():
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_probabilities.extend(probabilities.cpu().numpy().tolist())

predictions = np.asarray(all_predictions, dtype=int)
probabilities = np.asarray(all_probabilities)
accuracy = accuracy_score(true_labels, predictions)

report = classification_report(
    true_labels,
    predictions,
    labels=[0, 1],
    target_names=["Legitimate", "Phishing"],
    output_dict=True,
    zero_division=0,
)
matrix = confusion_matrix(true_labels, predictions, labels=[0, 1])

print("\n--- RESULTS ---")
print(f"Accuracy: {accuracy:.4%}")
print("\nClassification report:")
print(classification_report(
    true_labels,
    predictions,
    labels=[0, 1],
    target_names=["Legitimate", "Phishing"],
    digits=4,
    zero_division=0,
))
print("Confusion matrix (rows=true, columns=predicted):")
print("                 Predicted legitimate | Predicted phishing")
print(f"True legitimate: {matrix[0, 0]:>19} | {matrix[0, 1]:>17}")
print(f"True phishing:   {matrix[1, 0]:>19} | {matrix[1, 1]:>17}")

results = {
    "model": "DistilBERT",
    "test_dataset": str(TEST_FILE.relative_to(ROOT)),
    "model_directory": str(MODEL_DIR.relative_to(ROOT)),
    "test_samples": int(len(df)),
    "accuracy": float(accuracy),
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "class_mapping": {"0": "Legitimate", "1": "Phishing"},
    "evaluation_settings": {
        "batch_size": batch_size,
        "max_length": max_length,
    },
}

json_path = OUTPUT_DIR / "text_model_evaluation.json"
json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

predictions_df = pd.DataFrame({
    "text": texts,
    "true_label": true_labels,
    "predicted_label": predictions,
    "legitimate_probability": probabilities[:, 0],
    "phishing_probability": probabilities[:, 1],
})
csv_path = OUTPUT_DIR / "text_model_predictions.csv"
predictions_df.to_csv(csv_path, index=False)

print(f"\nSaved metrics: {json_path}")
print(f"Saved per-email predictions: {csv_path}")
print("Evaluation complete. No training was performed.")
