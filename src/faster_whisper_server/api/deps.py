from fastapi import File, HTTPException, UploadFile


async def valid_wav_file(
    file: UploadFile = File(..., description="WAV audio file"),
) -> UploadFile:
    """Dependency to validate that the uploaded file is a WAV format."""
    if file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="File must be a WAV audio format")

    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Filename must end with .wav")

    return file
