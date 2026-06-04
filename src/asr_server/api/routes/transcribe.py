from fastapi import APIRouter, Depends, UploadFile

from asr_server.api.deps import valid_wav_file
from asr_server.schemas.transcribe import (
    TranscribeRequestBody,
    TranscribeResponseBody,
)
from asr_server.service.transcribe import transcribe_upload

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
