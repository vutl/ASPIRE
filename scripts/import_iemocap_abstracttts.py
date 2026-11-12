"""Import the pinned AbstractTTS IEMOCAP audio and create a local 5-fold manifest."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import wave

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/IEMOCAP_HF_AbstractTTS/data"
DEST = ROOT / "datasets/IEMOCAP"
MANIFEST = ROOT / "datasets/IEMOCAP_KFOLD/tenfolds/all.csv"
LOCAL_MANIFEST = ROOT / "datasets/IEMOCAP_KFOLD/tenfolds/all.local.csv"
REPORT = ROOT / "datasets/IEMOCAP_HF_AbstractTTS/import_report.json"
REVISION = "9f1696a135a65ce997d898d4121c952269a822ca"
SHARDS = {
    "train-00000-of-00003.parquet": "8a37ed291467501fa1a689d2c6588b44c9d74a3e77e6c5a0338a008f50a95170",
    "train-00001-of-00003.parquet": "7ebfcb4b0d4f2aa7e6bdf9d81261b37b1c6bdb027750a2eefda0d3626ff19870",
    "train-00002-of-00003.parquet": "f1c1a321e5902304c8ef6280be615e0babbee7f0985a0e7ae88d95e69aad73af",
}
NAME_PATTERN = re.compile(r"^(Ses0[1-5][FM]_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)_([FM]\d+)\.wav$")


def audio_identity(stream):
    with wave.open(stream, "rb") as audio:
        params = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes())
        if min(params) <= 0 or audio.getcomptype() != "NONE":
            raise ValueError(f"Invalid or unsupported WAV: {params}")
        expected = params[0] * params[1] * params[3]
        data = audio.readframes(params[3])
        if len(data) != expected:
            raise ValueError(f"Truncated WAV: expected {expected} bytes, found {len(data)}")
        return params, hashlib.sha256(data).hexdigest()


def main():
    for name, expected in SHARDS.items():
        with (SOURCE / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {actual}")
    print("All three Parquet SHA-256 checks passed", flush=True)

    index = {}
    created = existing = 0
    conflicts = []
    failures = []
    for shard in SHARDS:
        parquet = pq.ParquetFile(SOURCE / shard)
        for batch in parquet.iter_batches(batch_size=128, columns=["file", "audio"]):
            for row in batch.to_pylist():
                name = row["file"]
                match = NAME_PATTERN.fullmatch(name or "")
                if not match or name in index:
                    raise ValueError(f"Unsafe, unexpected, or duplicate filename: {name!r}")
                session = int(name[3:5])
                target = DEST / f"Session{session}" / "sentences/wav" / match.group(1) / name
                payload = row["audio"]["bytes"] if row["audio"] else None
                try:
                    if not payload:
                        raise ValueError("Audio bytes are missing")
                    source_identity = audio_identity(io.BytesIO(payload))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        with target.open("rb") as stream:
                            local_identity = audio_identity(stream)
                        if source_identity != local_identity:
                            conflicts.append(str(target.relative_to(ROOT)))
                        existing += 1
                    else:
                        with target.open("xb") as stream:
                            stream.write(payload)
                        created += 1
                    index[name] = target
                except Exception as error:
                    failures.append({"file": name, "error": str(error)})
        print(f"{shard}: {len(index)} audio checked", flush=True)

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    missing = []
    for row in rows:
        name = row["sample_name"] + ".wav"
        if name not in index or not index[name].is_file():
            missing.append(name)
        else:
            row["path"] = str(index[name])

    report = {
        "source": "https://huggingface.co/datasets/AbstractTTS/IEMOCAP",
        "revision": REVISION,
        "parquet_sha256": SHARDS,
        "audio_count": len(index),
        "new_wavs": created,
        "existing_wavs": existing,
        "existing_pcm_conflicts": conflicts,
        "audio_failures": failures,
        "manifest_rows": len(rows),
        "manifest_missing": missing,
        "local_manifest": str(LOCAL_MANIFEST),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if conflicts or failures or missing or len(index) != 10039 or len(rows) != 5531:
        raise SystemExit("Import incomplete; local manifest was not written")
    if LOCAL_MANIFEST.exists():
        raise SystemExit(f"Refusing to overwrite existing manifest: {LOCAL_MANIFEST}")
    with LOCAL_MANIFEST.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {LOCAL_MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
