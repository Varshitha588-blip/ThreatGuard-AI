from pathlib import Path
from io import BytesIO

import pyarrow.parquet as pq
import soundfile as sf

shard = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--SpeechAntiSpoofingBenchmarks--CVoiceFake_small" / "snapshots" / "7d787a585e3af9a093698936c06441748f4c1a20" / "data" / "test-00000-of-00014.parquet"

parquet = pq.ParquetFile(shard)
found = {}

for batch in parquet.iter_batches(batch_size=32, columns=["path", "label", "audio"]):
    for row in batch.to_pylist():
        label = row["label"]
        if label not in found:
            found[label] = row

    if 0 in found and 1 in found:
        break

for label, name in [(0, "GENUINE"), (1, "SPOOF")]:
    row = found[label]
    audio = row["audio"]
    data, sample_rate = sf.read(BytesIO(audio["bytes"]))

    print(f"\n{name}")
    print("Path:", row["path"])
    print("Sample rate:", sample_rate, "Hz")
    print("Duration:", round(len(data) / sample_rate, 2), "seconds")
    print("Audio shape:", data.shape)
    print("Decode successful:", len(data) > 0)
