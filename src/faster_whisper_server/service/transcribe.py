import asyncio
import shutil
import tempfile
from pathlib import Path

from fastapi import UploadFile

from faster_whisper_server.schemas.transcribe import TranscribeResponseBody
from faster_whisper_server.service.whisper_model import sync_transcribe_file

# Single Whisper model / GPU — one transcription at a time (like the washing machine lock).
_transcribe_lock = asyncio.Lock()


async def transcribe_upload(file: UploadFile) -> TranscribeResponseBody:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        async with _transcribe_lock:
            return await asyncio.to_thread(sync_transcribe_file, tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
