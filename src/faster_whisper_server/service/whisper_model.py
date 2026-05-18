import os
from pathlib import Path

from faster_whisper import WhisperModel

from faster_whisper_server.loaders import cuda_utils  # noqa: F401 — register CUDA DLLs on import
from faster_whisper_server.schemas.transcribe import TranscribeResponseBody

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            os.getenv("WHISPER_MODEL", "base"),
            device=os.getenv("WHISPER_DEVICE", "cuda"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
        )
    return _model


def sync_transcribe_file(path: Path) -> TranscribeResponseBody:
    """Synchronous blocking transcription; run via asyncio.to_thread from the event loop."""
    segments_iter, info = _get_model().transcribe(
        str(path),
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    transcribed_text = "".join(seg.text for seg in segments_iter)
    return TranscribeResponseBody(
        transcribed_text=transcribed_text,
        language=info.language,
        language_probability=info.language_probability,
    )
