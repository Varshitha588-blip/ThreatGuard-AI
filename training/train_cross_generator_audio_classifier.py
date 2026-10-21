from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/audio/cross_generator/features_wav2vec2/cross_generator_embeddings.npz"
MODEL_DIR = ROOT / "models/audio"
REPORT_DIR = ROOT / "reports/audio"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading combined embeddings...")

with np.load(DATA_PATH, allow_pickle=False) as data:
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    splits = data["split"].astype(str)

train_mask = splits == "train"
val_mask = splits == "validation"
seen_mask = splits == "seen_test"
unseen_mask = splits == "unseen_test"

if not all(mask.any() for mask in [train_mask, val_mask, seen_mask, unseen_mask]):
    raise ValueError("One or more required dataset splits are empty.")

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Seen-generator test:", int(seen_mask.sum()))
print("Unseen-generator test:", int(unseen_mask.sum()))

def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "spoof_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "spoof_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "spoof_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            labels=[0, 1],
            target_names=["genuine", "spoof"],
            output_dict=True,
            zero_division=0,
        ),
    }

best_model = None
best_c = None
best_score = -1.0
validation_results = {}

for c in [0.01, 0.1, 1.0, 10.0]:
    print(f"\nTraining Logistic Regression (C={c})...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    result = metrics(y_val, predictions)
    validation_results[str(c)] = result

    print(
        f"Validation accuracy={result['accuracy']:.4f} | "
        f"Spoof precision={result['spoof_precision']:.4f} | "
        f"Spoof recall={result['spoof_recall']:.4f} | "
        f"Spoof F1={result['spoof_f1']:.4f}"
    )

    if result["spoof_f1"] > best_score:
        best_score = result["spoof_f1"]
        best_model = model
        best_c = c

print(f"\nSelected C={best_c} using validation spoof F1={best_score:.4f}")

model_path = MODEL_DIR / "cross_generator_wav2vec2_classifier.joblib"
joblib.dump(best_model, model_path)

# Final evaluation: do not use these test results to select or tune the model.
seen_results = metrics(y[seen_mask], best_model.predict(X[seen_mask]))
unseen_results = metrics(y[unseen_mask], best_model.predict(X[unseen_mask]))

report = {
    "embedding_model": "facebook/wav2vec2-base",
    "embedding_dimensions": int(X.shape[1]),
    "training_recordings": int(train_mask.sum()),
    "validation_recordings": int(val_mask.sum()),
    "seen_generator_test_recordings": int(seen_mask.sum()),
    "unseen_generator_test_recordings": int(unseen_mask.sum()),
    "classifier": "StandardScaler + LogisticRegression",
    "selected_C": best_c,
    "selection_metric": "validation spoof F1",
    "validation_candidates": validation_results,
    "seen_generator_test": seen_results,
    "unseen_generator_test": unseen_results,
}

report_path = REPORT_DIR / "cross_generator_audio_evaluation.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

for title, result in [
    ("SEEN-GENERATOR TEST", seen_results),
    ("UNSEEN-GENERATOR TEST", unseen_results),
]:
    print(f"\n=== {title} ===")
    print(f"Accuracy:        {result['accuracy']:.4f}")
    print(f"Spoof precision: {result['spoof_precision']:.4f}")
    print(f"Spoof recall:    {result['spoof_recall']:.4f}")
    print(f"Spoof F1:         {result['spoof_f1']:.4f}")
    print("Confusion matrix [[genuine correct, genuine called spoof],")
    print("                  [spoof called genuine, spoof correct]]:")
    print(np.array(result["confusion_matrix"]))
    print("\nClassification report:")
    print(classification_report(
        y[seen_mask] if title == "SEEN-GENERATOR TEST" else y[unseen_mask],
        best_model.predict(X[seen_mask] if title == "SEEN-GENERATOR TEST" else X[unseen_mask]),
        labels=[0, 1],
        target_names=["genuine", "spoof"],
        zero_division=0,
    ))

print("\nModel saved to:", model_path)
print("Evaluation report saved to:", report_path)
print("\nTRAINING AND EVALUATION COMPLETE")
