from pathlib import Path
import time

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModel

MODEL_NAME = "facebook/wav2vec2-base"
ROOT = Path.cwd()
MANIFEST = ROOT / "data" / "audio" / "audio_manifest.csv"
OUTPUT = ROOT / "data" / "audio" / "features" / "wav2vec2_embeddings.npz"


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST).reset_index(drop=True)

    if len(df) != 400:
        raise ValueError(f"Expected 400 recordings, found {len(df)}")

    if df["filepath"].duplicated().any():
        raise ValueError("Duplicate audio file paths found.")

    # Verify that no underlying utterance occurs in multiple splits.
    split_counts = df.groupby("group_id")["split"].nunique()
    if (split_counts > 1).any():
        raise ValueError("Data leakage: a group_id appears in multiple splits.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} on CPU...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    embeddings = []
    started = time.time()

    print(f"\nExtracting embeddings for {len(df)} recordings...")
    print("Keep the laptop plugged in; CPU processing may take a while.")

    for i, row in enumerate(df.itertuples(index=False), start=1):
        audio_path = ROOT / row.filepath

        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio: {audio_path}")

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        inputs = extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        vector = embedding.cpu().numpy().astype(np.float32)

        if vector.shape != (768,):
            raise ValueError(
                f"Unexpected embedding shape {vector.shape} for {audio_path.name}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(f"Invalid embedding values: {audio_path.name}")

        embeddings.append(vector)

        if i % 20 == 0 or i == len(df):
            elapsed = (time.time() - started) / 60
            print(f"Processed {i}/{len(df)} | elapsed: {elapsed:.1f} min")

    X = np.vstack(embeddings)

    np.savez_compressed(
        OUTPUT,
        X=X,
        labels=df["label"].to_numpy(dtype=np.int64),
        splits=df["split"].to_numpy(dtype=str),
        group_ids=df["group_id"].to_numpy(dtype=str),
        class_names=df["class_name"].to_numpy(dtype=str),
        filepaths=df["filepath"].to_numpy(dtype=str),
    )

    print("\n========== EMBEDDING SUMMARY ==========")
    print(f"Recordings: {len(df)}")
    print(f"Embedding matrix: {X.shape}")
    print(f"Finite values: {bool(np.isfinite(X).all())}")
    print(f"Saved to: {OUTPUT}")
    print(f"Total time: {(time.time() - started) / 60:.1f} minutes")
    print("SUCCESS: Wav2Vec2 embeddings saved.")


if __name__ == "__main__":
    main()
