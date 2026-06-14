"""Resolve and download iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common model weights."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download

MS_REPO_ID = "iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common"
CKPT_NAME = "pretrained_eres2netv2w24s4ep4.ckpt"
# Short local dir name under worker-root .models/ (see workers/registry.toml).
MODEL_DIR_NAME = "speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common"
_MIN_CKPT_BYTES = 1_000_000

_MODEL_RESOLUTION_LOCK = threading.Lock()


def get_models_dir() -> Path:
    """Worker project root .models/ (see workers/registry.toml)."""
    return Path(__file__).resolve().parent.parent.parent / ".models"


def local_model_dir() -> Path:
    return get_models_dir() / MODEL_DIR_NAME


def local_ckpt_path() -> Path:
    return local_model_dir() / CKPT_NAME


def is_model_ready(path: Path | None = None) -> bool:
    """Check that the SV checkpoint exists and is non-trivial in size."""
    ckpt_path = path or local_ckpt_path()
    if not ckpt_path.is_file():
        return False
    return ckpt_path.stat().st_size >= _MIN_CKPT_BYTES


def download_to_local() -> Path:
    """Download model into .models and return the checkpoint path."""
    model_dir = local_model_dir()
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {MS_REPO_ID} -> {model_dir}", file=sys.stderr)
    snapshot_download(MS_REPO_ID, local_dir=str(model_dir))
    ckpt_path = model_dir / CKPT_NAME
    if not is_model_ready(ckpt_path):
        raise RuntimeError(
            f"Downloaded model to {model_dir} but checkpoint is missing or incomplete "
            f"({CKPT_NAME})."
        )
    print(f"Done: {ckpt_path}", file=sys.stderr)
    return ckpt_path


def resolve_model_path() -> tuple[Path, str]:
    """Resolve checkpoint path with .models priority.

    Returns:
        (ckpt_path, source) where source is ``local`` or ``downloaded``.
    """
    ckpt_path = local_ckpt_path()
    if is_model_ready(ckpt_path):
        return ckpt_path, "local"
    return download_to_local(), "downloaded"


def ensure_model_path() -> tuple[Path, str]:
    """Thread-safe entry: resolve or download the checkpoint path."""
    with _MODEL_RESOLUTION_LOCK:
        return resolve_model_path()


def run_download() -> tuple[Path, str]:
    """CLI entry: ensure model is available and return path + source."""
    return ensure_model_path()
