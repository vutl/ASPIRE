import csv
import json
import os
import wave
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "datasets"
DATASETS = ["EMOVOME", "emovome_original", "IEMOCAP", "IEMOCAP_KFOLD", "MSP-Podcast"]
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        yield from csv.DictReader(f)


def wav_check(path):
    try:
        size = path.stat().st_size
        with wave.open(str(path), "rb") as w:
            meta = {
                "channels": w.getnchannels(), "sample_width": w.getsampwidth(),
                "rate": w.getframerate(), "frames": w.getnframes(),
                "compression": w.getcomptype(),
            }
            if min(meta["channels"], meta["sample_width"], meta["rate"], meta["frames"]) <= 0:
                raise ValueError(f"invalid metadata: {meta}")
            expected = meta["frames"] * meta["channels"] * meta["sample_width"]
            got = 0
            while True:
                block = w.readframes(65536)
                if not block:
                    break
                got += len(block)
            if meta["compression"] == "NONE" and got < expected:
                raise ValueError(f"truncated PCM: expected {expected} decoded bytes, got {got}")
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


all_audio = []
per_dataset = {}
basename_index = defaultdict(list)
relative_index = {}
for ds in DATASETS:
    base = ROOT / ds
    files = [p for p in base.rglob("*") if p.is_file()]
    audio = [p for p in files if p.suffix.lower() in AUDIO_EXTS]
    wavs = [p for p in audio if p.suffix.lower() == ".wav"]
    per_dataset[ds] = {
        "files": len(files), "audio": len(audio), "wav": len(wavs),
        "audio_extensions": dict(Counter(p.suffix.lower() for p in audio)),
    }
    all_audio.extend(audio)
    for p in audio:
        basename_index[p.name.lower()].append(p)
        relative_index[p.relative_to(ROOT).as_posix().lower()] = p

corrupt = []
for i, p in enumerate(all_audio, 1):
    if p.suffix.lower() == ".wav":
        error = wav_check(p)
        if error:
            corrupt.append({"path": str(p.relative_to(ROOT)), "error": error})


def resolve(raw, manifest):
    raw = (raw or "").strip().strip('"').replace("\\", "/")
    if not raw:
        return "empty", None
    low = raw.lower()
    candidates = [low]
    if "datasets/" in low:
        candidates.append(low.split("datasets/", 1)[1])
    for candidate in candidates:
        if candidate in relative_index:
            return "exact", relative_index[candidate]
    local = (manifest.parent / raw)
    if local.is_file():
        return "exact", local
    hits = basename_index.get(Path(raw).name.lower(), [])
    if len(hits) == 1:
        return "basename", hits[0]
    if len(hits) > 1:
        return "ambiguous", None
    return "missing", None


manifest_specs = [
    (ROOT / "EMOVOME/vutl_split/train.csv", "audio_path"),
    (ROOT / "EMOVOME/vutl_split/val.csv", "audio_path"),
    (ROOT / "EMOVOME/vutl_split/test.csv", "audio_path"),
    (ROOT / "IEMOCAP_KFOLD/all.csv", "path"),
    (ROOT / "MSP-Podcast/Labels/labels_consensus.csv", "FileName"),
]
manifests = []
all_referenced = set()
for manifest, column in manifest_specs:
    result = Counter()
    examples = defaultdict(list)
    values = []
    for row in rows(manifest):
        raw = row.get(column, "")
        values.append(raw)
        status, found = resolve(raw, manifest)
        result[status] += 1
        if found:
            all_referenced.add(found.resolve())
        elif len(examples[status]) < 20:
            examples[status].append(raw)
    manifests.append({
        "manifest": str(manifest.relative_to(ROOT)), "column": column,
        "rows": len(values), "unique_values": len(set(values)),
        "resolution": dict(result), "problem_examples": dict(examples),
    })

# Original EMOVOME uses file_id rather than an explicit path.
for manifest in [ROOT / "emovome_original/labels.csv", ROOT / "emovome_original/transcriptions.csv"]:
    result = Counter()
    examples = defaultdict(list)
    values = []
    for row in rows(manifest):
        raw = row.get("file_id", "")
        values.append(raw)
        names = [raw] if Path(raw).suffix else [raw + ".wav"]
        hits = []
        for name in names:
            hits.extend(basename_index.get(Path(name).name.lower(), []))
        hits = [p for p in hits if "emovome_original" in p.parts]
        status = "exact" if len(hits) == 1 else "ambiguous" if len(hits) > 1 else "missing"
        result[status] += 1
        if len(hits) == 1:
            all_referenced.add(hits[0].resolve())
        elif len(examples[status]) < 20:
            examples[status].append(raw)
    manifests.append({
        "manifest": str(manifest.relative_to(ROOT)), "column": "file_id",
        "rows": len(values), "unique_values": len(set(values)),
        "resolution": dict(result), "problem_examples": dict(examples),
    })

unreferenced = [str(p.relative_to(ROOT)) for p in all_audio if p.resolve() not in all_referenced]
report = {
    "root": str(ROOT), "per_dataset": per_dataset,
    "total_audio": len(all_audio), "wav_decode_errors": corrupt,
    "manifests": manifests,
    "unreferenced_audio_count": len(unreferenced),
    "unreferenced_audio_examples": unreferenced[:50],
}
out = Path(__file__).resolve().parent / "dataset_audit_report.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
