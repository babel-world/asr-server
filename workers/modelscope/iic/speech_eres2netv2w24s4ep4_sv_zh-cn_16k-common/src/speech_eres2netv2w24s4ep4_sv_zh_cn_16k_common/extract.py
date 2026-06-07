"""Extract SV embeddings from in-memory waveforms (production) or .npy files (CLI tests).

This worker is tuned for the GPT-SoVITS v2Pro pipeline B (SV / speaker embedding).
Callers are expected to supply waveforms that upstream has already prepared:
mono float32 @ 16 kHz, conventional amplitude in [-1, 1] (no 1145.14 or 32768 scaling).

This module does not resample or re-normalize — those stay in the glue layer.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from speech_eres2netv2w24s4ep4_sv_zh_cn_16k_common import download

TARGET_SAMPLE_RATE = 16_000
OUTPUT_DIM = 20480
NUM_MEL_BINS = 80

ERES2NET_DIR = Path(__file__).resolve().parent / "vendor" / "eres2net"

_runtime_lock = threading.Lock()
_runtime: SvRuntime | None = None


@dataclass
class SvRuntime:
    """Cached ERes2NetV2 model and inference settings."""

    ckpt_path: Path
    device: torch.device
    use_fp16: bool
    model: torch.nn.Module


def _ensure_eres2net_path() -> None:
    path = str(ERES2NET_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def _ensure_kaldi():
    _ensure_eres2net_path()
    import kaldi as Kaldi  # noqa: WPS433

    return Kaldi


def normalize_waveform(arr: np.ndarray) -> np.ndarray:
    """Accept (T,) or (1, T); return float32 mono waveform (T,)."""
    waveform = np.asarray(arr, dtype=np.float32)
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        return waveform.reshape(-1)
    raise ValueError(f"Expected waveform shape (T,) or (1, T), got {waveform.shape}")


def resolve_device() -> tuple[torch.device, bool]:
    """Return (device, use_fp16). FP16 enabled only on CUDA."""
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def _load_model(ckpt_path: Path, device: torch.device, use_fp16: bool) -> torch.nn.Module:
    _ensure_eres2net_path()
    from ERes2NetV2 import ERes2NetV2  # noqa: WPS433

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
    model.load_state_dict(state)
    model.eval()

    if use_fp16:
        model = model.half()
    return model.to(device)


def get_runtime() -> SvRuntime:
    """Return cached runtime; reload when ckpt path or device changes."""
    global _runtime

    ckpt_path, _source = download.ensure_model_path()
    device, use_fp16 = resolve_device()

    with _runtime_lock:
        if (
            _runtime is not None
            and _runtime.ckpt_path == ckpt_path
            and _runtime.device == device
            and _runtime.use_fp16 == use_fp16
        ):
            return _runtime

        model = _load_model(ckpt_path, device, use_fp16)
        _runtime = SvRuntime(
            ckpt_path=ckpt_path,
            device=device,
            use_fp16=use_fp16,
            model=model,
        )
        return _runtime


def extract_sv_single(waveform: np.ndarray) -> np.ndarray:
    """Extract SV embedding for one 16 kHz mono waveform; output (1, 20480) float32."""
    wav = normalize_waveform(waveform)
    runtime = get_runtime()
    Kaldi = _ensure_kaldi()

    wav_t = torch.from_numpy(wav).to(runtime.device)
    if runtime.use_fp16:
        wav_t = wav_t.half()

    with torch.no_grad():
        feat = torch.stack(
            [
                Kaldi.fbank(
                    wav_t.unsqueeze(0),
                    num_mel_bins=NUM_MEL_BINS,
                    sample_frequency=TARGET_SAMPLE_RATE,
                    dither=0,
                )
            ]
        )
        sv_emb = runtime.model.forward3(feat)

    out = sv_emb.detach().cpu().float().numpy().astype(np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    return out


def extract_sv_many(waveforms: list[np.ndarray]) -> list[np.ndarray]:
    """Extract SV embeddings sequentially; each output is (1, 20480) float32."""
    return [extract_sv_single(waveform) for waveform in waveforms]


def load_sv_input_npy(input_path: Path) -> np.ndarray:
    """Load mono float32 16 kHz waveform from .npy; return 1D (T,)."""
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
    """CLI/test helper: sv_input.npy -> sv_output.npy (1, 20480)."""
    waveform = load_sv_input_npy(input_path)
    features = extract_sv_single(waveform)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)
    return output_path
