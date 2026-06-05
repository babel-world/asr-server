"""Resolve faster-whisper model paths across .models and Hugging Face cache."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from faster_whisper import download_model
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

_MODEL_RESOLUTION_LOCK = threading.Lock()

# Must-have core files for a valid CTranslate2 faster-whisper model.
_CT2_CORE_FILES = (
    "model.bin",
    "config.json",
)
_CT2_VOCAB_CANDIDATES = ("vocabulary.txt", "vocabulary.json", "tokenizer.json")


def get_repo_root() -> Path:
    """Project root (directory containing pyproject.toml)."""
    env_root = os.getenv("ASR_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # src/asr_server/utils/whisper_assets.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def get_local_models_dir() -> Path:
    """Directory for project-local model storage (default: <repo>/.models)."""
    custom = os.getenv("WHISPER_MODELS_DIR")
    if custom:
        return Path(custom).resolve()
    return get_repo_root() / ".models"


def hf_repo_id(model_size: str) -> str:
    return f"Systran/faster-whisper-{model_size}"


def local_model_dir(model_size: str) -> Path:
    return get_local_models_dir() / f"faster-whisper-{model_size}"


def is_custom_model_path(model_name: str) -> bool:
    """True when WHISPER_MODEL points to an existing directory with model weights."""
    path = Path(model_name)
    return path.is_dir() and is_ct2_model_dir(path)


def is_ct2_model_dir(path: Path) -> bool:
    """Check that a directory contains a complete CTranslate2 faster-whisper model."""
    if not all((path / name).is_file() for name in _CT2_CORE_FILES):
        return False
    return any((path / name).is_file() for name in _CT2_VOCAB_CANDIDATES)


def find_hf_snapshot(model_size: str) -> Path | None:
    """Return a valid HF hub snapshot path via official local cache API, if any."""
    try:
        snapshot_path = snapshot_download(
            hf_repo_id(model_size),
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return None
    except OSError:
        return None

    resolved = Path(snapshot_path).resolve()
    if not is_ct2_model_dir(resolved):
        return None
    return resolved


def resolve_whisper_model_path(model_size: str) -> tuple[Path, str]:
    """Resolve model path with .models priority over HF cache.

    Returns:
        (resolved_path, source) where source is one of:
        "local", "hf_cache", or "downloaded".
    """
    local_path = local_model_dir(model_size)
    local_valid = is_ct2_model_dir(local_path)
    hf_path = find_hf_snapshot(model_size)
    hf_valid = hf_path is not None

    if local_valid and hf_valid:
        return local_path, "local"
    if local_valid:
        return local_path, "local"
    if hf_valid and hf_path is not None:
        return hf_path, "hf_cache"

    return _download_to_local(model_size), "downloaded"


def _download_to_local(model_size: str) -> Path:
    """Download model into .models and return the local directory path."""
    local_path = local_model_dir(model_size)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    download_model(model_size, output_dir=str(local_path))
    if not is_ct2_model_dir(local_path):
        raise RuntimeError(
            f"Downloaded faster-whisper model to {local_path} but it is incomplete "
            f"(missing model.bin, config.json, or valid vocabulary files)."
        )
    return local_path


def ensure_whisper_model_path(model_name: str) -> tuple[Path, str]:
    """Thread-safe entry: resolve model path for WHISPER_MODEL env value."""
    if is_custom_model_path(model_name):
        return Path(model_name).resolve(), "custom"

    with _MODEL_RESOLUTION_LOCK:
        return resolve_whisper_model_path(model_name)
