import json
import time
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
MANIFEST = ROOT / "data" / "audio" / "audio_manifest.csv"
MODEL_DIR = ROOT / "models" / "audio"
REPORT_DIR = ROOT / "reports" / "audio"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def extract_features(filepath):
    """Convert one audio recording into a fixed-length numeric feature vector."""
    y, sr = librosa.load(filepath, sr=16000, mono=True)

    if len(y) < 512:
        raise ValueError(f"Audio is too short: {filepath}")

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=20, n_fft=512, hop_length=160
    )
    delta = librosa.feature.delta(mfcc, axis=1)

    feature_parts = [
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.std(delta, axis=1),
    ]

    extra_features = [
        librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=512, hop_length=160
        ),
        librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=512, hop_length=160
        ),
        librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=512, hop_length=160
        ),
        librosa.feature.zero_crossing_rate(
            y, frame_length=512, hop_length=160
        ),
        librosa.feature.rms(y=y, frame_length=512, hop_length=160),
    ]

    for feature in extra_features:
        feature_parts.extend([
            np.array([np.mean(feature)]),
            np.array([np.std(feature)]),
        ])

    result = np.concatenate(feature_parts).astype(np.float32)

    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite features found: {filepath}")

    return result


def main():
    started = time.time()

    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)

    required = {"filepath", "label", "split", "group_id", "class_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    df["label"] = df["label"].astype(int)

    # Check that each underlying utterance occurs in only one split.
    split_counts = df.groupby("group_id")["split"].nunique()
    if (split_counts > 1).any():
        raise ValueError(
            "Data leakage detected: some group IDs occur in multiple splits."
        )

    print(f"Manifest records: {len(df)}")
    print("Extracting audio features. This may take several minutes...")

    features = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        audio_path = ROOT / row.filepath

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file missing: {audio_path}")

        features.append(extract_features(audio_path))

        if i % 50 == 0 or i == len(df):
            print(f"Processed {i}/{len(df)} recordings")

    X = np.vstack(features)
    y = df["label"].to_numpy()
    splits = df["split"].to_numpy()

    print(f"Feature matrix shape: {X.shape}")

    train_mask = splits == "train"
    val_mask = splits == "validation"
    test_mask = splits == "test"

    if not (train_mask.any() and val_mask.any() and test_mask.any()):
        raise ValueError("Train, validation, and test splits must all exist.")

    candidates = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    best_name = None
    best_score = -1.0

    print("\nTraining and comparing models on validation data...")

    for name, model in candidates.items():
        model.fit(X[train_mask], y[train_mask])
        val_predictions = model.predict(X[val_mask])
        val_f1 = f1_score(y[val_mask], val_predictions, zero_division=0)

        results[name] = {
            "validation_accuracy": float(
                accuracy_score(y[val_mask], val_predictions)
            ),
            "validation_precision": float(
                precision_score(y[val_mask], val_predictions, zero_division=0)
            ),
            "validation_recall": float(
                recall_score(y[val_mask], val_predictions, zero_division=0)
            ),
            "validation_f1": float(val_f1),
        }

        print(f"\n{name}")
        print(f"  Validation accuracy: {results[name]['validation_accuracy']:.4f}")
        print(f"  Validation precision: {results[name]['validation_precision']:.4f}")
        print(f"  Validation recall: {results[name]['validation_recall']:.4f}")
        print(f"  Validation F1: {val_f1:.4f}")

        if val_f1 > best_score:
            best_score = val_f1
            best_name = name

    # Refit the selected model on training + validation data.
    train_val_mask = train_mask | val_mask
    best_model = candidates[best_name]
    best_model.fit(X[train_val_mask], y[train_val_mask])

    test_predictions = best_model.predict(X[test_mask])

    test_results = {
        "selected_model": best_name,
        "test_accuracy": float(
            accuracy_score(y[test_mask], test_predictions)
        ),
        "test_precision": float(
            precision_score(y[test_mask], test_predictions, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y[test_mask], test_predictions, zero_division=0)
        ),
        "test_f1": float(
            f1_score(y[test_mask], test_predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y[test_mask], test_predictions, labels=[0, 1]
        ).tolist(),
        "validation_results": results,
        "recordings": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "feature_count": int(X.shape[1]),
        "label_mapping": {"0": "genuine", "1": "spoof"},
        "warning": (
            "Preliminary baseline trained on a 400-recording sample. "
            "Results do not establish generalization to unseen spoofing systems."
        ),
    }

    model_path = MODEL_DIR / "audio_spoof_baseline.joblib"
    report_path = REPORT_DIR / "audio_baseline_report.json"

    joblib.dump({
        "model": best_model,
        "feature_count": int(X.shape[1]),
        "sample_rate": 16000,
        "label_mapping": {0: "genuine", 1: "spoof"},
    }, model_path)

    report_path.write_text(
        json.dumps(test_results, indent=2), encoding="utf-8"
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Selected model: {best_name}")
    print(f"Test accuracy:  {test_results['test_accuracy']:.4f}")
    print(f"Test precision: {test_results['test_precision']:.4f}")
    print(f"Test recall:    {test_results['test_recall']:.4f}")
    print(f"Test F1:        {test_results['test_f1']:.4f}")
    print("\nConfusion matrix (rows=true, columns=predicted; genuine, spoof):")
    print(np.array(test_results["confusion_matrix"]))
    print("\nDetailed classification report:")
    print(classification_report(
        y[test_mask],
        test_predictions,
        labels=[0, 1],
        target_names=["genuine", "spoof"],
        zero_division=0,
    ))
    print(f"Model saved to: {model_path}")
    print(f"Report saved to: {report_path}")
    print(f"Total time: {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
