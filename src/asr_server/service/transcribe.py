import asyncio
import shutil
import tempfile
from pathlib import Path

from fastapi import UploadFile

from asr_server.schemas.transcribe import (
    TranscribeModelStateResponseBody,
    TranscribeResponseBody,
)
from asr_server.service.whisper_model import (
    release_model,
    sync_transcribe_file,
    warmup_model,
)

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


async def transcribe_start() -> TranscribeModelStateResponseBody:
    async with _transcribe_lock:
        newly_loaded = await asyncio.to_thread(warmup_model)
    if newly_loaded:
        return TranscribeModelStateResponseBody(
            loaded=True,
            message="Whisper model loaded.",
        )
    return TranscribeModelStateResponseBody(
        loaded=True,
        message="Whisper model was already loaded.",
    )


async def transcribe_stop() -> TranscribeModelStateResponseBody:
    async with _transcribe_lock:
        released = await asyncio.to_thread(release_model)
    if released:
        return TranscribeModelStateResponseBody(
            loaded=False,
            message="Whisper model released.",
        )
    return TranscribeModelStateResponseBody(
        loaded=False,
        message="Whisper model was not loaded.",
    )
