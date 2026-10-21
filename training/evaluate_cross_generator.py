from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/audio/cross_generator/features_wav2vec2/cross_generator_embeddings.npz"
MODEL_PATH = ROOT / "models/audio/cross_generator_wav2vec2_classifier.joblib"
REPORT_DIR = ROOT / "reports/audio"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("THREATGUARD AI - AUDIO MODEL EVALUATION")
print("=" * 65)

with np.load(DATA_PATH, allow_pickle=False) as data:
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    splits = data["split"].astype(str)
    system_ids = data["system_id"].astype(str)
    filepaths = data["filepath"].astype(str)

model = joblib.load(MODEL_PATH)
print(f"Loaded classifier: {MODEL_PATH.name}")
print(f"Embedding shape: {X.shape}")
print(f"Total recordings: {len(y)}")

results = {}
prediction_rows = []

for split_name in ["seen_test", "unseen_test"]:
    mask = splits == split_name

    if not mask.any():
        raise ValueError(f"No recordings found for split: {split_name}")

    X_test = X[mask]
    y_true = y[mask]
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        classes = list(model.classes_)
        spoof_column = classes.index(1)
        spoof_probabilities = probabilities[:, spoof_column]
    else:
        spoof_probabilities = np.full(len(y_true), np.nan)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["genuine", "spoof"],
        output_dict=True,
        zero_division=0,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    results[split_name] = {
        "recordings": int(mask.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "genuine_precision": float(precision[0]),
        "genuine_recall": float(recall[0]),
        "genuine_f1": float(f1[0]),
        "spoof_precision": float(precision[1]),
        "spoof_recall": float(recall[1]),
        "spoof_f1": float(f1[1]),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    print(f"\n--- {split_name.upper()} ---")
    print(f"Recordings:      {int(mask.sum())}")
    print(f"Accuracy:        {accuracy_score(y_true, y_pred):.4%}")
    print(f"Spoof precision: {precision[1]:.4f}")
    print(f"Spoof recall:    {recall[1]:.4f}")
    print(f"Spoof F1:        {f1[1]:.4f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print("                 Predicted genuine | Predicted spoof")
    print(f"True genuine:    {matrix[0, 0]:>16} | {matrix[0, 1]:>14}")
    print(f"True spoof:      {matrix[1, 0]:>16} | {matrix[1, 1]:>14}")
    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["genuine", "spoof"],
        digits=4,
        zero_division=0,
    ))

    indices = np.flatnonzero(mask)
    for local_idx, global_idx in enumerate(indices):
        prediction_rows.append({
            "split": split_name,
            "filepath": filepaths[global_idx],
            "system_id": system_ids[global_idx],
            "true_label": int(y_true[local_idx]),
            "predicted_label": int(y_pred[local_idx]),
            "true_class": "spoof" if y_true[local_idx] == 1 else "genuine",
            "predicted_class": "spoof" if y_pred[local_idx] == 1 else "genuine",
            "spoof_probability": float(spoof_probabilities[local_idx]),
        })

report_data = {
    "embedding_model": "facebook/wav2vec2-base",
    "classifier": "StandardScaler + LogisticRegression",
    "embedding_dimensions": int(X.shape[1]),
    "model_path": str(MODEL_PATH.relative_to(ROOT)),
    "dataset_path": str(DATA_PATH.relative_to(ROOT)),
    "label_mapping": {"0": "genuine", "1": "spoof"},
    "test_results": results,
    "caution": (
        "Results apply to these dataset splits and do not guarantee "
        "performance on all real-world voices or deepfake generators."
    ),
}

report_path = REPORT_DIR / "cross_generator_audio_evaluation.json"
report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

import csv
predictions_path = REPORT_DIR / "cross_generator_audio_predictions.csv"
with predictions_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=prediction_rows[0].keys())
    writer.writeheader()
    writer.writerows(prediction_rows)

print("\nSaved evaluation report:", report_path)
print("Saved test predictions:", predictions_path)
print("Evaluation complete. No training was performed.")
