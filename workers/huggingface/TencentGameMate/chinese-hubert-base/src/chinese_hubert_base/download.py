"""Resolve and download TencentGameMate/chinese-hubert-base model assets."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

HF_REPO_ID = "TencentGameMate/chinese-hubert-base"
HF_TOKEN_ENV = "HF_TOKEN"
MODEL_DIR_NAME = "chinese-hubert-base"
_WEIGHT_FILE = "pytorch_model.bin"
# Real weights are ~hundreds of MB; smaller files indicate placeholders or incomplete downloads.
_MIN_WEIGHT_BYTES = 1_000_000

_MODEL_RESOLUTION_LOCK = threading.Lock()


def hf_token() -> str | None:
    """Return HF_TOKEN from the environment, or None when unset/empty."""
    return os.getenv(HF_TOKEN_ENV, "") or None


def get_models_dir() -> Path:
    """Worker-local model storage: <package>/.models."""
    return Path(__file__).resolve().parent / ".models"


def local_model_dir() -> Path:
    return get_models_dir() / MODEL_DIR_NAME


def is_model_dir_complete(path: Path) -> bool:
    """Check config.json and a non-trivial pytorch_model.bin exist."""
    config = path / "config.json"
    weights = path / _WEIGHT_FILE
    if not config.is_file() or not weights.is_file():
        return False
    return weights.stat().st_size >= _MIN_WEIGHT_BYTES


def find_hf_snapshot_local_only() -> Path | None:
    """Return a valid HF hub snapshot path via local cache only, if any."""
    try:
        snapshot_path = snapshot_download(
            HF_REPO_ID,
            local_files_only=True,
            token=hf_token(),
        )
    except LocalEntryNotFoundError:
        return None
    except OSError:
        return None

    resolved = Path(snapshot_path).resolve()
    if not is_model_dir_complete(resolved):
        return None
    return resolved


def download_to_local() -> Path:
    """Download model into .models and return the local directory path."""
    local_path = local_model_dir()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {HF_REPO_ID} -> {local_path}")
    snapshot_download(
        HF_REPO_ID,
        local_dir=str(local_path),
        token=hf_token(),
    )
    if not is_model_dir_complete(local_path):
        raise RuntimeError(
            f"Downloaded model to {local_path} but it is incomplete "
            f"(missing config.json or valid {_WEIGHT_FILE})."
        )
    print(f"Done: {local_path}")
    return local_path


def resolve_model_path() -> tuple[Path, str]:
    """Resolve model path with .models priority over HF cache.

    Returns:
        (resolved_path, source) where source is one of:
        "local", "hf_cache", or "downloaded".
    """
    local_path = local_model_dir()
    local_valid = is_model_dir_complete(local_path)
    hf_path = find_hf_snapshot_local_only()
    hf_valid = hf_path is not None

    if local_valid:
        print(f"Already present, skipping: {local_path}")
        return local_path, "local"
    if hf_valid:
        return hf_path, "hf_cache"

    return download_to_local(), "downloaded"


def ensure_model_path() -> tuple[Path, str]:
    """Thread-safe entry: resolve or download the model path."""
    with _MODEL_RESOLUTION_LOCK:
        return resolve_model_path()


def run_download() -> tuple[Path, str]:
    """CLI entry: ensure model is available and return path + source."""
    return ensure_model_path()
