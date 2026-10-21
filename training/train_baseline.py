
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "phishing_messages.csv"
OUT = ROOT / "models" / "text"
REPORTS = ROOT / "reports"


def main():
    df = pd.read_csv(DATA)

    if df["label"].nunique() < 2:
        raise ValueError("Training requires both classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["LEGITIMATE", "PHISHING"],
        zero_division=0,
        output_dict=True,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, OUT / "baseline.joblib")

    with open(REPORTS / "baseline_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["LEGITIMATE", "PHISHING"],
        zero_division=0,
    ))
    print("Model saved to models/text/baseline.joblib")


if __name__ == "__main__":
    main()
