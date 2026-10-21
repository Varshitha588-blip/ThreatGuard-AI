from pathlib import Path

import librosa
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModel

MODEL_NAME = "facebook/wav2vec2-base"
ROOT = Path.cwd()
MANIFEST = ROOT / "data" / "audio" / "audio_manifest.csv"

def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)

    # Test two genuine and two spoof recordings from the training split.
    sample = (
        df[df["split"] == "train"]
        .groupby("class_name", group_keys=False)
        .head(2)
        .reset_index(drop=True)
    )

    if len(sample) != 4 or set(sample["class_name"]) != {"genuine", "spoof"}:
        raise ValueError("Could not select two recordings from each class.")

    print(f"Loading pretrained model: {MODEL_NAME}")
    print("Device: CPU")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    print("\nModel loaded successfully.")
    print("Extracting embeddings from four recordings...\n")

    for row in sample.itertuples(index=False):
        audio_path = ROOT / row.filepath
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file missing: {audio_path}")

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        inputs = extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # Average the time dimension to obtain one vector per recording.
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)

        print(f"File: {audio_path.name}")
        print(f"Actual class: {row.class_name}")
        print(f"Embedding shape: {tuple(embedding.shape)}")
        print(f"Finite values: {bool(torch.isfinite(embedding).all())}")
        print("-" * 45)

    print("\nSUCCESS: Wav2Vec2 embedding extraction works.")

if __name__ == "__main__":
    main()
