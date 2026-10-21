
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    Path.home()
    / ".cache" / "huggingface" / "hub"
    / "datasets--SpeechAntiSpoofingBenchmarks--CVoiceFake_small"
    / "snapshots" / "7d787a585e3af9a093698936c06441748f4c1a20"
    / "data"
)

SHARDS = [
    DATA_DIR / f"test-{i:05d}-of-00014.parquet"
    for i in range(3)
]

MANIFEST_PATH = (
    ROOT / "data" / "audio" / "cross_generator"
    / "cross_generator_manifest.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "audio" / "cross_generator" / "recordings"
)

OUTPUT_MANIFEST = (
    ROOT / "data" / "audio" / "cross_generator"
    / "extracted_manifest.csv"
)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    if manifest["source_path"].duplicated().any():
        raise ValueError("Duplicate source paths in manifest.")

    expected = {
        row.source_path: row
        for row in manifest.itertuples(index=False)
    }

    # Do not silently reuse a previous partial extraction.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.rglob("*.mp3")):
        raise FileExistsError(
            f"MP3 files already exist in {OUTPUT_DIR}. "
            "Inspect the folder before rerunning."
        )

    for shard in SHARDS:
        if not shard.exists():
            raise FileNotFoundError(shard)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extracted = []
    found = set()
    total_bytes = 0

    print("Expected recordings:", len(expected))
    print("Output folder:", OUTPUT_DIR)

    for shard in SHARDS:
        print(f"\nScanning {shard.name}...")
        parquet = pq.ParquetFile(shard)

        for batch in parquet.iter_batches(
            columns=["path", "audio"],
            batch_size=128,
        ):
            paths = batch.column("path").to_pylist()
            audios = batch.column("audio").to_pylist()

            for source_path, audio in zip(paths, audios):
                if source_path not in expected:
                    continue

                if source_path in found:
                    raise RuntimeError(
                        f"Duplicate source path: {source_path}"
                    )

                if not audio or not audio.get("bytes"):
                    raise RuntimeError(
                        f"Missing audio bytes: {source_path}"
                    )

                row = expected[source_path]
                filename = Path(source_path).name

                # Preserve split and class separation.
                destination = (
                    OUTPUT_DIR / row.split / row.class_name / filename
                )
                destination.parent.mkdir(
                    parents=True, exist_ok=True
                )
                destination.write_bytes(audio["bytes"])

                found.add(source_path)
                total_bytes += len(audio["bytes"])

                extracted.append({
                    "base_id": row.base_id,
                    "system_id": row.system_id,
                    "language": row.language,
                    "label": row.label,
                    "class_name": row.class_name,
                    "split": row.split,
                    "source_path": source_path,
                    "filepath": str(destination.relative_to(ROOT)),
                    "audio_bytes": len(audio["bytes"]),
                })

        print(f"Matched and saved: {len(found)} / {len(expected)}")

    missing = set(expected) - found
    if missing:
        raise RuntimeError(
            f"Extraction incomplete: {len(missing)} recordings missing. "
            f"Examples: {list(sorted(missing))[:5]}"
        )

    result = pd.DataFrame(extracted)

    # Final validation before writing the manifest.
    if len(result) != len(manifest):
        raise RuntimeError("Extracted row count does not match manifest.")

    result.to_csv(OUTPUT_MANIFEST, index=False)

    print("\n=== EXTRACTION COMPLETE ===")
    print("Recordings:", len(result))
    print("Audio size (MB):", round(total_bytes / 1e6, 2))
    print("\nCounts by split and class:")
    print(result.groupby(["split", "class_name"]).size().to_string())
    print("\nExtracted manifest:", OUTPUT_MANIFEST)


if __name__ == "__main__":
    main()
