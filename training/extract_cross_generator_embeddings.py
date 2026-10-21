
from pathlib import Path
import time

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModel


ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    ROOT / "data" / "audio" / "cross_generator"
    / "extracted_manifest.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "audio" / "cross_generator"
    / "features_wav2vec2" / "chunks"
)

MODEL_NAME = "facebook/wav2vec2-base"
SAMPLE_RATE = 16000
EMBEDDING_DIM = 768
CHUNK_SIZE = 50


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST).reset_index(drop=True)

    if len(df) != 14360:
        raise ValueError(f"Expected 14360 rows, got {len(df)}")

    if df["filepath"].duplicated().any():
        raise ValueError("Duplicate audio file paths detected.")

    # Check that the same base utterance never crosses dataset splits.
    split_counts = df.groupby("base_id")["split"].nunique()
    if (split_counts > 1).any():
        raise ValueError("Data leakage: base IDs occur across splits.")

    audio_paths = [ROOT / p for p in df["filepath"]]
    missing = [p for p in audio_paths if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio files are missing. "
            f"Examples: {missing[:5]}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset rows: {len(df)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Output: {OUTPUT_DIR}")
    print("Loading Wav2Vec2 on CPU...")

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    total_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    started = time.time()
    completed = 0

    for chunk_number, start in enumerate(
        range(0, len(df), CHUNK_SIZE)
    ):
        end = min(start + CHUNK_SIZE, len(df))
        chunk_path = OUTPUT_DIR / f"chunk_{chunk_number:04d}.npz"

        expected_paths = df.iloc[start:end]["filepath"].to_numpy(
            dtype=str
        )

        # Resume safely if this exact chunk was already completed.
        if chunk_path.exists():
            with np.load(chunk_path, allow_pickle=False) as saved:
                valid = (
                    int(saved["start"]) == start
                    and int(saved["end"]) == end
                    and np.array_equal(
                        saved["filepaths"], expected_paths
                    )
                    and saved["X"].shape
                    == (end - start, EMBEDDING_DIM)
                    and np.isfinite(saved["X"]).all()
                )

            if valid:
                completed += 1
                continue

            raise RuntimeError(
                f"Existing chunk is invalid: {chunk_path}. "
                "Inspect it before deleting or replacing it."
            )

        vectors = []

        for index in range(start, end):
            audio_path = audio_paths[index]

            audio, sr = librosa.load(
                audio_path,
                sr=SAMPLE_RATE,
                mono=True,
            )

            if audio.size == 0:
                raise ValueError(f"Empty audio file: {audio_path}")

            inputs = extractor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
            )

            with torch.inference_mode():
                outputs = model(**inputs.to(device))

            embedding = (
                outputs.last_hidden_state
                .mean(dim=1)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if embedding.shape != (EMBEDDING_DIM,):
                raise ValueError(
                    f"Unexpected embedding shape for {audio_path}: "
                    f"{embedding.shape}"
                )

            if not np.isfinite(embedding).all():
                raise ValueError(
                    f"Non-finite embedding for {audio_path}"
                )

            vectors.append(embedding)

        X_chunk = np.vstack(vectors)

        # Write a temporary file first so interrupted writes do not
        # leave a seemingly complete chunk behind.
        temporary_path = chunk_path.with_suffix(".tmp.npz")

        np.savez_compressed(
            temporary_path,
            X=X_chunk,
            start=np.array(start),
            end=np.array(end),
            filepaths=expected_paths,
        )
        temporary_path.replace(chunk_path)

        completed += 1
        elapsed = time.time() - started

        print(
            f"Chunks: {completed}/{total_chunks} | "
            f"Recordings processed this run: {end}/{len(df)} | "
            f"Elapsed: {elapsed / 60:.1f} min"
        )

    print("\n=== EMBEDDING EXTRACTION COMPLETE ===")
    print("Total recordings:", len(df))
    print("Embedding dimensions:", EMBEDDING_DIM)
    print("Chunks:", total_chunks)
    print("Chunk folder:", OUTPUT_DIR)
    print("Elapsed this run: {:.1f} minutes".format(
        (time.time() - started) / 60
    ))


if __name__ == "__main__":
    main()
