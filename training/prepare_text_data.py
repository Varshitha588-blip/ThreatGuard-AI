
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "phishing_messages.csv"
OUTPUT = ROOT / "data" / "processed" / "phishing_messages.csv"


def main():
    df = pd.read_csv(INPUT)

    required = {"text", "label", "source"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    df = df.dropna(subset=["text", "label", "source"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="raise")

    df = df[df["text"].str.len() > 0]
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset=["text"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Saved {len(df)} messages to {OUTPUT}")
    print("\nLabel counts:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
