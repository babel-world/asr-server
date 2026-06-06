"""Extract HuBERT features from GPT-SoVITS-style waveform .npy inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import HubertModel

from chinese_hubert_base import download

TARGET_SAMPLE_RATE = 16_000


def load_waveform_npy(input_path: Path) -> np.ndarray:
    """Load mono float32 waveform; accept (T,) or (1, T), return 1D (T,)."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    waveform = np.load(input_path)
    if not isinstance(waveform, np.ndarray):
        raise TypeError(f"Expected np.ndarray in {input_path}, got {type(waveform)!r}")

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        return waveform.reshape(-1)
    raise ValueError(
        f"Expected waveform shape (T,) or (1, T), got {waveform.shape} in {input_path}"
    )


def resolve_device() -> tuple[torch.device, bool]:
    """Return (device, use_fp16). FP16 is enabled only on CUDA."""
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def run_extract(input_path: Path, output_path: Path) -> Path:
    """Extract last_hidden_state features and save as float32 .npy (1, T, 768)."""
    model_path, _source = download.ensure_model_path()
    
    # 1. 直接加载波形
    waveform = load_waveform_npy(input_path) 
    device, use_fp16 = resolve_device()

    # 2. 仅加载 Model 本干，不加载 Extractor
    model = HubertModel.from_pretrained(str(model_path)).to(device).eval()
    if use_fp16:
        model = model.half()

    # 3. 将一维波形提升为 [1, T] 丢进 GPU
    input_values = torch.from_numpy(waveform).unsqueeze(0).to(device)
    if use_fp16:
        input_values = input_values.half()

    # 4. 直接前向传播
    with torch.no_grad():
        last_hidden_state = model(input_values).last_hidden_state

    # 5. 保持 [1, T, 768] 落盘，把 .transpose 留给 Pipeline 处理！
    features = last_hidden_state.detach().cpu().float().numpy().astype(np.float32)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)
    return output_path
