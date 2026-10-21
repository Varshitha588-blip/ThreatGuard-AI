
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed" / "transformer"

DATASET_NAME = "ansulev/phishing-email-dataset"
TEXT_COLUMN = "Email Text"
LABEL_COLUMN = "Email Type"

# Adjust these only after inspecting the real label values.
MAX_CHARS = 20000


def map_label(value):
    label = str(value).strip().lower()

    if "phishing" in label:
        return 1

    if any(word in label for word in ["safe", "legitimate", "benign", "ham"]):
        return 0

    raise ValueError(
        f"Unknown label: {value!r}. Inspect the dataset labels first."
    )


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    df = dataset["train"].to_pandas()

    print("Original shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Missing text column: {TEXT_COLUMN}")

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COLUMN}")

    print("\nOriginal label values:")
    print(df[LABEL_COLUMN].value_counts(dropna=False))

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    df.columns = ["text", "original_label"]

    df = df.dropna(subset=["text", "original_label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    # Avoid pathological text lengths and excessive memory usage.
    df = df[df["text"].str.len() <= MAX_CHARS]

    df["label"] = df["original_label"].apply(map_label)
    df = df[["text", "label"]]

    # Identical messages must not appear in multiple splits.
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("Expected both legitimate and phishing classes.")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, part in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        path = OUTPUT_DIR / f"{name}.csv"
        part.to_csv(path, index=False)

        print(f"\n{name}: {len(part)} rows")
        print(part["label"].value_counts().sort_index())
        print(f"Saved: {path}")

    print("\nPreparation complete.")
    print("Split sizes:", len(train_df), len(val_df), len(test_df))


if __name__ == "__main__":
    main()
