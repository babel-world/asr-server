from fastapi import APIRouter, Depends, UploadFile

from asr_server.api.deps import valid_wav_file
from asr_server.schemas.transcribe import (
    TranscribeModelStateResponseBody,
    TranscribeRequestBody,
    TranscribeResponseBody,
)
from asr_server.service.transcribe import (
    transcribe_start,
    transcribe_stop,
    transcribe_upload,
)

router = APIRouter(prefix="/transcribe", tags=["transcribe"])


@router.post(
    "",
    response_model=TranscribeResponseBody,
    summary=TranscribeRequestBody.__doc__ or "",
)
async def transcribe(
    file: UploadFile = Depends(valid_wav_file),
) -> TranscribeResponseBody:
    return await transcribe_upload(file)


@router.post(
    "/start",
    response_model=TranscribeModelStateResponseBody,
    summary="Load faster-whisper model into memory.",
)
async def transcribe_model_start() -> TranscribeModelStateResponseBody:
    return await transcribe_start()


@router.post(
    "/stop",
    response_model=TranscribeModelStateResponseBody,
    summary="Release faster-whisper model from memory.",
)
async def transcribe_model_stop() -> TranscribeModelStateResponseBody:
    return await transcribe_stop()
