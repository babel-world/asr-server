"""Extract HuBERT features from in-memory waveforms (production) or .npy files (CLI tests)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import HubertModel

from chinese_hubert_base import download

TARGET_SAMPLE_RATE = 16_000
HIDDEN_SIZE = 768


@dataclass
class HubertRuntime:
    """Cached HuBERT model and inference settings."""

    model_path: Path
    device: torch.device
    use_fp16: bool
    model: HubertModel


_runtime_lock = threading.Lock()
_runtime: HubertRuntime | None = None
_model_load_count = 0


def normalize_waveform(arr: np.ndarray) -> np.ndarray:
    """Accept (T,) or (1, T); return float32 mono waveform (T,)."""
    waveform = np.asarray(arr, dtype=np.float32)
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        return waveform.reshape(-1)
    raise ValueError(f"Expected waveform shape (T,) or (1, T), got {waveform.shape}")


def resolve_device() -> tuple[torch.device, bool]:
    """Return (device, use_fp16). FP16 is enabled only on CUDA."""
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def _load_model(model_path: Path, device: torch.device, use_fp16: bool) -> HubertModel:
    global _model_load_count
    model = HubertModel.from_pretrained(str(model_path)).to(device).eval()
    if use_fp16:
        model = model.half()
    _model_load_count += 1
    return model


def get_runtime() -> HubertRuntime:
    """Return a cached runtime; reload when resolved model path changes."""
    global _runtime
    model_path, _source = download.ensure_model_path()
    device, use_fp16 = resolve_device()

    with _runtime_lock:
        if (
            _runtime is not None
            and _runtime.model_path == model_path
            and _runtime.device == device
            and _runtime.use_fp16 == use_fp16
        ):
            return _runtime

        model = _load_model(model_path, device, use_fp16)
        _runtime = HubertRuntime(
            model_path=model_path,
            device=device,
            use_fp16=use_fp16,
            model=model,
        )
        return _runtime


def get_model_load_count() -> int:
    """Number of times the model weights were loaded (for tests)."""
    return _model_load_count


def extract_features_single(waveform: np.ndarray) -> np.ndarray:
    """Extract features for one waveform; output shape (1, T', 768) float32."""
    wav = normalize_waveform(waveform)
    runtime = get_runtime()

    input_values = torch.from_numpy(wav).unsqueeze(0).to(runtime.device)
    if runtime.use_fp16:
        input_values = input_values.half()

    with torch.no_grad():
        last_hidden_state = runtime.model(input_values).last_hidden_state

    return last_hidden_state.detach().cpu().float().numpy().astype(np.float32)


def extract_features_many(waveforms: list[np.ndarray]) -> list[np.ndarray]:
    """Extract features for multiple waveforms sequentially (not tensor batching).

    Each output is (1, T', 768) float32; lengths may differ per item.
    """
    return [extract_features_single(waveform) for waveform in waveforms]


def load_waveform_npy(input_path: Path) -> np.ndarray:
    """Load mono float32 waveform from .npy; return 1D (T,)."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    waveform = np.load(input_path)
    if not isinstance(waveform, np.ndarray):
        raise TypeError(f"Expected np.ndarray in {input_path}, got {type(waveform)!r}")

    try:
        return normalize_waveform(waveform)
    except ValueError as exc:
        raise ValueError(f"{exc} in {input_path}") from exc


def run_extract(input_path: Path, output_path: Path) -> Path:
    """CLI/test helper: npy in -> npy out via in-memory extraction."""
    waveform = load_waveform_npy(input_path)
    features = extract_features_single(waveform)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)
    return output_path
