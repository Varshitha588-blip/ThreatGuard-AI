from pathlib import Path
import json
import time

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "audio" / "features" / "wav2vec2_embeddings.npz"
MODEL_PATH = ROOT / "models" / "audio" / "wav2vec2_spoof_classifier.joblib"
REPORT_PATH = ROOT / "reports" / "audio" / "wav2vec2_report.json"


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spoof": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_spoof": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_spoof": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {DATA_PATH}"
        )

    data = np.load(DATA_PATH, allow_pickle=False)

    X = data["X"]
    y = data["labels"]
    splits = data["splits"]
    group_ids = data["group_ids"]

    if X.shape != (400, 768):
        raise ValueError(f"Unexpected embedding shape: {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError("Embeddings contain invalid values.")

    # Confirm every utterance remains in only one split.
    for group_id in np.unique(group_ids):
        group_splits = np.unique(splits[group_ids == group_id])
        if len(group_splits) != 1:
            raise ValueError(
                f"Data leakage detected for group {group_id}"
            )

    train_mask = splits == "train"
    val_mask = splits == "validation"
    test_mask = splits == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("========== DATA SUMMARY ==========")
    print(f"Training:   {len(y_train)} recordings")
    print(f"Validation: {len(y_val)} recordings")
    print(f"Testing:    {len(y_test)} recordings")
    print("Label 0 = genuine | Label 1 = spoof")
    print(f"Training labels: {np.bincount(y_train)}")
    print(f"Validation labels: {np.bincount(y_val)}")
    print(f"Test labels: {np.bincount(y_test)}")

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=10000,
                random_state=42,
            )),
        ]),
    }

    results = {}
    best_name = None
    best_model = None
    best_f1 = -1.0

    print("\n========== VALIDATION RESULTS ==========")

    for name, model in candidates.items():
        started = time.time()
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_val)
        metrics = calculate_metrics(y_val, val_predictions)
        results[name] = metrics

        print(f"\n{name}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision_spoof']:.4f}")
        print(f"Recall:    {metrics['recall_spoof']:.4f}")
        print(f"F1-score:  {metrics['f1_spoof']:.4f}")
        print(f"Time:      {time.time() - started:.1f} seconds")

        if metrics["f1_spoof"] > best_f1:
            best_f1 = metrics["f1_spoof"]
            best_name = name
            best_model = model

    # Evaluate the selected model on the test split.
    # The test split is not used to choose the model.
    test_predictions = best_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_predictions)

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Selected model: {best_name}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision_spoof']:.4f}")
    print(f"Recall:    {test_metrics['recall_spoof']:.4f}")
    print(f"F1-score:  {test_metrics['f1_spoof']:.4f}")
    print("\nConfusion matrix (rows=true, columns=predicted)")
    print("Order: genuine, spoof")
    print(np.array(test_metrics["confusion_matrix"]))
    print("\nDetailed classification report:")
    print(classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["genuine", "spoof"],
        zero_division=0,
    ))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    report = {
        "model_type": "Wav2Vec2 embeddings + classical classifier",
        "embedding_model": "facebook/wav2vec2-base",
        "embedding_shape": list(X.shape),
        "selected_classifier": best_name,
        "label_mapping": {"0": "genuine", "1": "spoof"},
        "split_sizes": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "validation_results": results,
        "test_results": test_metrics,
        "limitations": [
            "Small experimental dataset of 400 recordings.",
            "Test metrics may vary with a larger and more diverse dataset.",
            "Results do not establish production-level deepfake detection.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Report saved to: {REPORT_PATH}")
    print("SUCCESS: Wav2Vec2 classifier training completed.")


if __name__ == "__main__":
    main()
