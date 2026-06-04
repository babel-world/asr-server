import gc
import os
from pathlib import Path

from faster_whisper import WhisperModel

from asr_server.loaders import cuda_utils  # noqa: F401 — register CUDA DLLs on import
from asr_server.schemas.transcribe import TranscribeResponseBody

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


def is_model_loaded() -> bool:
    return _model is not None


def warmup_model() -> bool:
    """Load WhisperModel into memory if not already loaded. Returns True if newly loaded."""
    already_loaded = is_model_loaded()
    _get_model()
    return not already_loaded


def release_model() -> bool:
    """Drop cached WhisperModel from memory. Returns True if a model was released."""
    global _model
    if _model is None:
        return False
    model = _model
    _model = None
    del model
    gc.collect()
    return True


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
