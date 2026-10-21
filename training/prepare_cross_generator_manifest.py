
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    Path.home()
    / ".cache" / "huggingface" / "hub"
    / "datasets--SpeechAntiSpoofingBenchmarks--CVoiceFake_small"
    / "snapshots" / "7d787a585e3af9a093698936c06441748f4c1a20"
    / "data"
)

SHARDS = [
    DATA_DIR / "test-00000-of-00014.parquet",
    DATA_DIR / "test-00001-of-00014.parquet",
    DATA_DIR / "test-00002-of-00014.parquet",
]

OUTPUT_DIR = ROOT / "data" / "audio" / "cross_generator"
OUTPUT_PATH = OUTPUT_DIR / "cross_generator_manifest.csv"

SEED = 42
TRAIN_SYSTEM = "vctk_parallel_wavegan.v1_generated"
HELD_OUT_SYSTEM = "vctk_multi_band_melgan.v2_generated"


def parse_notes(value):
    if isinstance(value, str):
        return json.loads(value)
    return value or {}


def base_id_from_path(audio_path):
    base = Path(audio_path).stem
    if base.endswith("_Gen"):
        base = base[:-4]
    return base


def main():
    for shard in SHARDS:
        if not shard.exists():
            raise FileNotFoundError(f"Missing shard: {shard}")

    rows = []

    for shard in SHARDS:
        print(f"Reading metadata: {shard.name}")
        table = pq.read_table(shard, columns=["path", "notes"])

        for audio_path, notes_value in zip(
            table["path"].to_pylist(),
            table["notes"].to_pylist(),
        ):
            notes = parse_notes(notes_value)
            system = notes.get("system_id", "UNKNOWN")
            language = notes.get("language", "UNKNOWN")

            if system not in {
                "bonafide",
                TRAIN_SYSTEM,
                HELD_OUT_SYSTEM,
            }:
                continue

            base_id = base_id_from_path(audio_path)
            label = 0 if system == "bonafide" else 1

            rows.append({
                "base_id": base_id,
                "system_id": system,
                "language": language,
                "label": label,
                "class_name": "genuine" if label == 0 else "spoof",
                "source_path": audio_path,
            })

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["base_id", "system_id"]
    )

    if df.empty:
        raise RuntimeError("No eligible recordings found.")

    genuine_ids = set(
        df.loc[df["system_id"] == "bonafide", "base_id"]
    )
    train_generator_ids = set(
        df.loc[df["system_id"] == TRAIN_SYSTEM, "base_id"]
    )
    held_out_ids = set(
        df.loc[df["system_id"] == HELD_OUT_SYSTEM, "base_id"]
    )


    # First select utterance IDs that have all three recordings:
    # genuine, seen-generator spoof, and held-out-generator spoof.
    common_ids = sorted(
        genuine_ids & train_generator_ids & held_out_ids
    )

    if len(common_ids) < 100:
        raise RuntimeError(
            f"Too few common utterance IDs: {len(common_ids)}"
        )

    # Split IDs before selecting recordings. This prevents
    # the same underlying utterance from leaking across splits.
    train_ids, remaining_ids = train_test_split(
        common_ids,
        test_size=0.30,
        random_state=SEED,
    )

    validation_ids, test_ids = train_test_split(
        remaining_ids,
        test_size=2 / 3,
        random_state=SEED,
    )

    seen_test_ids, unseen_test_ids = train_test_split(
        test_ids,
        test_size=0.50,
        random_state=SEED,
    )

    split_ids = {
        "train": set(train_ids),
        "validation": set(validation_ids),
        "seen_test": set(seen_test_ids),
        "unseen_test": set(unseen_test_ids),
    }


    parts = []

    for split, ids in split_ids.items():
        if split == "unseen_test":
            chosen_systems = {"bonafide", HELD_OUT_SYSTEM}
        else:
            chosen_systems = {"bonafide", TRAIN_SYSTEM}

        part = df[
            df["base_id"].isin(ids)
            & df["system_id"].isin(chosen_systems)
        ].copy()

        part["split"] = split
        parts.append(part)

    manifest = pd.concat(parts, ignore_index=True)

    # Verify no utterance ID crosses split boundaries.
    split_counts = manifest.groupby("base_id")["split"].nunique()
    if (split_counts > 1).any():
        raise RuntimeError("Leakage detected across splits.")

    # Verify both classes are present in every split.
    class_counts = manifest.groupby(["split", "class_name"]).size()
    for split in split_ids:
        for class_name in ["genuine", "spoof"]:
            if class_counts.get((split, class_name), 0) == 0:
                raise RuntimeError(
                    f"Missing class {class_name!r} in split {split!r}"
                )

    # Verify the held-out generator is absent from seen splits.
    seen_rows = manifest[manifest["split"] != "unseen_test"]
    if (seen_rows["system_id"] == HELD_OUT_SYSTEM).any():
        raise RuntimeError("Held-out generator leaked into seen splits.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_PATH, index=False)

    print("\nRows by split and class:")
    print(manifest.groupby(["split", "class_name"]).size().to_string())

    print("\nRows by split and generator:")
    print(manifest.groupby(["split", "system_id"]).size().to_string())

    print("\nUnique utterance IDs by split:")
    print(manifest.groupby("split")["base_id"].nunique().to_string())

    print("\nLeakage check: PASSED")
    print("Held-out generator:", HELD_OUT_SYSTEM)
    print("Manifest saved to:", OUTPUT_PATH)
    print("Metadata only; no audio files or models were changed.")


if __name__ == "__main__":
    main()
