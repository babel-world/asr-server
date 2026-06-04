import asyncio
import csv
import tempfile
from pathlib import Path

from fastapi import UploadFile

from asr_server.service.transcribe import _transcribe_lock
from asr_server.service.whisper_model import sync_transcribe_file
from asr_server.utils.manifest_zip import (
    collect_slice_wavs_from_dir,
    resolve_wav_path,
    safe_extract_zip,
)

MANIFEST_FIELDS = ("filename", "speaker", "language", "text", "probability")


def sync_build_manifest_from_zip(zip_bytes: bytes) -> tuple[Path, Path, str]:
    """从切片 ZIP 生成 manifest CSV。返回 ``(csv_path, session_dir, download_filename)``。"""
    session_dir = Path(tempfile.mkdtemp(prefix="asr_manifest_"))
    extract_dir = session_dir / "extracted"
    extract_dir.mkdir()

    zip_path = session_dir / "upload.zip"
    zip_path.write_bytes(zip_bytes)
    safe_extract_zip(zip_path, extract_dir)

    entries = collect_slice_wavs_from_dir(extract_dir)
    base_name = entries[0].base_name
    speaker = base_name

    csv_path = session_dir / f"{base_name}_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for entry in entries:
            wav_path = resolve_wav_path(extract_dir, entry)
            result = sync_transcribe_file(wav_path)
            writer.writerow(
                {
                    "filename": entry.filename,
                    "speaker": speaker,
                    "language": result.language,
                    "text": result.transcribed_text,
                    "probability": result.language_probability,
                }
            )

    download_name = f"{base_name}_manifest.csv"
    return csv_path, session_dir, download_name


async def build_manifest_upload(file: UploadFile) -> tuple[Path, Path, str]:
    zip_bytes = await file.read()
    async with _transcribe_lock:
        return await asyncio.to_thread(sync_build_manifest_from_zip, zip_bytes)
