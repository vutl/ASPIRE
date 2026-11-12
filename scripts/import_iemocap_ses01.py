"""Import and validate the Session 1 WAVs from 2notprime/IEMOCAP."""
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "datasets/IEMOCAP_HF_2notprime/ses01.zip"
DEST = ROOT / "datasets/IEMOCAP/Session1/sentences/wav"
MANIFEST = ROOT / "datasets/IEMOCAP_KFOLD/tenfolds/all.csv"
EXPECTED_SHA256 = "3a676853b8c82da6bcc6e483b2e00aa876deb869541109b3706533985e9c83db"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if digest(ARCHIVE) != EXPECTED_SHA256:
        raise ValueError("Archive SHA-256 does not match the Hugging Face revision")
    extracted = existing = 0
    conflicts = []
    failures = []
    imported = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive failed its CRC check")
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if len(parts) != 3 or parts[0] != "ses01" or not parts[2].lower().endswith(".wav"):
                continue
            if any(p in (".", "..") or ":" in p or "\\" in p for p in parts):
                raise ValueError(f"Unsafe archive path: {info.filename}")
            target = DEST / parts[1] / parts[2]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with archive.open(info) as source:
                    archive_hash = hashlib.file_digest(source, "sha256").hexdigest()
                if digest(target) != archive_hash:
                    conflicts.append(str(target.relative_to(ROOT)))
                    continue
                existing += 1
            else:
                with archive.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                extracted += 1
            imported.append(target)
            try:
                with sf.SoundFile(target) as sound:
                    if sound.samplerate <= 0 or sound.channels <= 0 or len(sound) <= 0:
                        raise ValueError("Invalid audio metadata")
                    frames = sum(len(block) for block in sound.blocks(blocksize=65536, dtype="float32"))
                    if frames != len(sound):
                        raise ValueError(f"Decoded {frames} frames, expected {len(sound)}")
            except Exception as error:
                failures.append({"path": str(target.relative_to(ROOT)), "error": str(error)})

    index = {p.name: p for p in imported}
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["fold"]) in (1, 2)]
    missing = [r["sample_name"] for r in rows if PurePosixPath(r["path"]).name not in index]
    report = {
        "source": "https://huggingface.co/datasets/2notprime/IEMOCAP",
        "revision": "f0bb4e89b9d49bbc3d13929e21681a9024695b2b",
        "archive_sha256": EXPECTED_SHA256,
        "new_wavs": extracted,
        "existing_matching_wavs": existing,
        "conflicting_existing_paths": conflicts,
        "decode_failures": failures,
        "session1_manifest_rows": len(rows),
        "session1_manifest_missing": missing,
    }
    report_path = ROOT / "datasets/IEMOCAP_HF_2notprime/ses01_import_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if conflicts or failures or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
