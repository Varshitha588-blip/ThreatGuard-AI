
import csv
import io
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent

PARQUET_PATH = (
    Path.home()
    / ".cache" / "huggingface" / "hub"
    / "datasets--SpeechAntiSpoofingBenchmarks--CVoiceFake_small"
    / "snapshots"
    / "7d787a585e3af9a093698936c06441748f4c1a20"
    / "data" / "test-00000-of-00014.parquet"
)

OUTPUT_DIR = ROOT / "data" / "audio" / "processed_large"
MANIFEST_PATH = ROOT / "data" / "audio" / "audio_manifest_large.csv"

SAMPLE_PAIRS = 1000
RANDOM_SEED = 42


def get_base_id(audio_path):
    filename = Path(audio_path).stem

    if filename.endswith("_Gen"):
        filename = filename[:-4]

    return filename


def main():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset shard not found: {PARQUET_PATH}"
        )

    print("Step 1: Reading audio metadata...")

    parquet_file = pq.ParquetFile(PARQUET_PATH)
    metadata = parquet_file.read(columns=["path", "label"])

    groups = defaultdict(dict)

    for audio_path, label in zip(
        metadata["path"].to_pylist(),
        metadata["label"].to_pylist(),
    ):
        base_id = get_base_id(audio_path)
        groups[base_id][int(label)] = audio_path

    matched = [
        (base_id, records)
        for base_id, records in groups.items()
        if 0 in records and 1 in records
    ]

    print("Available matched pairs:", len(matched))

    if len(matched) < SAMPLE_PAIRS:
        raise ValueError(
            f"Need {SAMPLE_PAIRS} matched pairs, "
            f"but found {len(matched)}."
        )

    random.Random(RANDOM_SEED).shuffle(matched)
    selected = matched[:SAMPLE_PAIRS]

    train_end = int(SAMPLE_PAIRS * 0.70)
    val_end = int(SAMPLE_PAIRS * 0.85)

    split_groups = {}

    for index, (base_id, records) in enumerate(selected):
        if index < train_end:
            split = "train"
        elif index < val_end:
            split = "validation"
        else:
            split = "test"

        split_groups[base_id] = {
            "split": split,
            "records": records,
        }

    selected_paths = {}

    for base_id, item in split_groups.items():
        for label, audio_path in item["records"].items():
            selected_paths[audio_path] = {
                "base_id": base_id,
                "label": label,
                "split": item["split"],
            }

    print("Selected pairs:", len(split_groups))
    print(
        f"Planned pairs — train: {train_end}, "
        f"validation: {val_end - train_end}, "
        f"test: {SAMPLE_PAIRS - val_end}"
    )

    for split in ("train", "validation", "test"):
        for class_name in ("genuine", "spoof"):
            (OUTPUT_DIR / split / class_name).mkdir(
                parents=True, exist_ok=True
            )

    manifest_rows = []
    found_paths = set()
    decode_errors = []

    print("\nStep 2: Extracting 2000 recordings to WAV...")
    print("Please keep the laptop plugged in.")

    for batch in parquet_file.iter_batches(
        batch_size=16,
        columns=["path", "label", "audio"],
    ):
        for row in batch.to_pylist():
            audio_path = row["path"]

            if audio_path not in selected_paths:
                continue

            if audio_path in found_paths:
                continue

            info = selected_paths[audio_path]

            try:
                audio_record = row["audio"]

                if not audio_record or not audio_record.get("bytes"):
                    raise ValueError("Audio bytes are missing")

                audio, sample_rate = sf.read(
                    io.BytesIO(audio_record["bytes"]),
                    dtype="float32",
                )

                if audio.ndim == 2:
                    audio = audio.mean(axis=1)

                if len(audio) == 0 or sample_rate <= 0:
                    raise ValueError("Audio is empty or invalid")

                class_name = (
                    "genuine" if info["label"] == 0 else "spoof"
                )

                filename = f"{info['base_id']}__{class_name}.wav"

                output_path = (
                    OUTPUT_DIR
                    / info["split"]
                    / class_name
                    / filename
                )

                sf.write(
                    str(output_path),
                    audio,
                    sample_rate,
                    subtype="PCM_16",
                )

                manifest_rows.append({
                    "filepath": str(
                        output_path.relative_to(ROOT)
                    ).replace("\\", "/"),
                    "label": info["label"],
                    "class_name": class_name,
                    "group_id": info["base_id"],
                    "split": info["split"],
                    "source_path": audio_path,
                    "sample_rate": sample_rate,
                    "duration_seconds": round(
                        len(audio) / sample_rate, 3
                    ),
                })

                found_paths.add(audio_path)

            except Exception as exc:
                decode_errors.append(f"{audio_path}: {exc}")

        print(
            f"Extracted {len(found_paths)} / "
            f"{len(selected_paths)} recordings"
        )

        if len(found_paths) == len(selected_paths):
            break

    print("\nStep 3: Writing the large manifest...")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filepath",
        "label",
        "class_name",
        "group_id",
        "split",
        "source_path",
        "sample_rate",
        "duration_seconds",
    ]

    with MANIFEST_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    counts = defaultdict(lambda: defaultdict(int))

    for row in manifest_rows:
        counts[row["split"]][row["class_name"]] += 1

    print("\n========== LARGE DATASET SUMMARY ==========")
    print("Selected pairs:", len(split_groups))
    print("Successfully extracted:", len(manifest_rows))
    print("Expected recordings:", SAMPLE_PAIRS * 2)
    print("Decode errors:", len(decode_errors))

    for split in ("train", "validation", "test"):
        print(
            f"{split}: genuine={counts[split]['genuine']}, "
            f"spoof={counts[split]['spoof']}"
        )

    print("Manifest:", MANIFEST_PATH)
    print("Audio output:", OUTPUT_DIR)

    if decode_errors:
        print("\nFirst decoding errors:")
        for error in decode_errors[:10]:
            print(error)

    if len(manifest_rows) != SAMPLE_PAIRS * 2:
        raise RuntimeError(
            "Dataset is incomplete. Do not train yet; "
            "check the summary and decode errors."
        )

    print("\nSUCCESS: Large audio dataset preparation completed.")


if __name__ == "__main__":
    main()

