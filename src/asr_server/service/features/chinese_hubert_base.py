"""chinese-hubert-base feature extraction (npy in -> npy out)."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np
from fastapi import UploadFile

from asr_server.infra.worker.session import get_worker_session
from asr_server.schemas.features import FeaturesModelStateResponseBody

WORKER_ALIAS = "chinese-hubert-base"

# Single chinese-hubert-base worker session — one extract at a time.
_features_lock = asyncio.Lock()


def format_features_download_name(source_filename: str) -> str:
    """Download filename: ``{stem}_features.npy``."""
    stem = Path(source_filename).stem or "waveform"
    return f"{stem}_features.npy"


def validate_waveform_npy(path: Path) -> None:
    """Ensure input is float32 mono waveform with shape (T,) or (1, T)."""
    waveform = np.load(path)
    if not isinstance(waveform, np.ndarray):
        raise ValueError(f"Expected np.ndarray in {path.name}, got {type(waveform)!r}")

    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim == 1:
        return
    if arr.ndim == 2 and arr.shape[0] == 1:
        return
    raise ValueError(
        f"Expected waveform shape (T,) or (1, T), got {waveform.shape}"
    )


def _session():
    return get_worker_session(WORKER_ALIAS)


def sync_start_session() -> FeaturesModelStateResponseBody:
    result = _session().start()
    if result.newly_started:
        return FeaturesModelStateResponseBody(
            loaded=True,
            message="chinese-hubert-base worker loaded.",
        )
    return FeaturesModelStateResponseBody(
        loaded=True,
        message="chinese-hubert-base worker was already loaded.",
    )


def sync_stop_session() -> FeaturesModelStateResponseBody:
    released = _session().stop()
    if released:
        return FeaturesModelStateResponseBody(
            loaded=False,
            message="chinese-hubert-base worker released.",
        )
    return FeaturesModelStateResponseBody(
        loaded=False,
        message="chinese-hubert-base worker was not loaded.",
    )


def sync_extract_npy(
    file_bytes: bytes,
    source_filename: str,
) -> tuple[Path, Path, str]:
    """Write upload to temp npy, extract via persistent worker session."""
    session_dir = Path(tempfile.mkdtemp(prefix="features-chinese-hubert-base-"))
    input_path = session_dir / "input.npy"
    output_path = session_dir / "output.npy"
    input_path.write_bytes(file_bytes)

    try:
        validate_waveform_npy(input_path)
    except ValueError:
        raise
    except OSError as e:
        raise ValueError(f"Failed to read uploaded npy: {e}") from e

    _session().extract_npy(input_path, output_path)

    if not output_path.is_file():
        raise RuntimeError("Worker finished but output.npy was not created")

    download_name = format_features_download_name(source_filename)
    return output_path, session_dir, download_name


async def extract_upload(
    file: UploadFile,
) -> tuple[Path, Path, str]:
    """Extract chinese-hubert-base features from uploaded waveform npy."""
    file_bytes = await file.read()
    filename = file.filename or "waveform.npy"

    async with _features_lock:
        return await asyncio.to_thread(
            sync_extract_npy,
            file_bytes,
            filename,
        )


async def features_start() -> FeaturesModelStateResponseBody:
    async with _features_lock:
        return await asyncio.to_thread(sync_start_session)


async def features_stop() -> FeaturesModelStateResponseBody:
    async with _features_lock:
        return await asyncio.to_thread(sync_stop_session)
