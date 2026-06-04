"""对 TEMP 中的音频执行静音切片，并将片段保存为 WAV。

用法（项目根目录）::

    uv run python scripts/test_slice.py
    uv run python scripts/test_slice.py TEMP/manbo.mp3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from asr_server.utils.audio import (
    calculate_audio_rms,
    find_silence_boundaries,
    normalize_audio,
    slice_waveform_by_boundaries,
)

TARGET_SR = 32_000

# 与原先 slice 服务默认参数一致
DEFAULT_THRESHOLD_DB = -40.0
DEFAULT_MIN_LENGTH_MS = 5000
DEFAULT_MIN_INTERVAL_MS = 300
DEFAULT_HOP_SIZE_MS = 20
DEFAULT_MAX_SIL_KEPT_MS = 5000


def _ms_to_frame_params(
    *,
    min_length_ms: int,
    min_interval_ms: int,
    hop_size_ms: int,
    max_sil_kept_ms: int,
) -> tuple[int, int, int, int, int]:
    hop_size_samples = round(TARGET_SR * hop_size_ms / 1000)
    if hop_size_samples <= 0:
        raise ValueError("hop_size_ms 过小，换算后的 hop_size_samples 必须为正整数")

    win_size_samples = min(
        round(TARGET_SR * min_interval_ms / 1000), 4 * hop_size_samples
    )
    if win_size_samples <= 0:
        raise ValueError("min_interval_ms 过小，换算后的 frame_length 必须为正整数")

    min_length_frames = round(TARGET_SR * min_length_ms / 1000 / hop_size_samples)
    min_interval_frames = round(TARGET_SR * min_interval_ms / 1000 / hop_size_samples)
    max_sil_kept_frames = round(TARGET_SR * max_sil_kept_ms / 1000 / hop_size_samples)
    return (
        hop_size_samples,
        win_size_samples,
        min_length_frames,
        min_interval_frames,
        max_sil_kept_frames,
    )


def slice_and_save(
    input_path: Path,
    output_dir: Path,
    *,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    min_length_ms: int = DEFAULT_MIN_LENGTH_MS,
    min_interval_ms: int = DEFAULT_MIN_INTERVAL_MS,
    hop_size_ms: int = DEFAULT_HOP_SIZE_MS,
    max_sil_kept_ms: int = DEFAULT_MAX_SIL_KEPT_MS,
) -> list[Path]:
    """读取音频、切片，并写入 ``{源文件名}_{start}_{end}.wav``。"""
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_name = input_path.name

    raw_waveform, source_sr = sf.read(input_path, dtype="float32")
    raw_waveform = np.asarray(raw_waveform, dtype=np.float32)

    clean_audio = normalize_audio(
        raw_waveform, source_sr=int(source_sr), target_sr=TARGET_SR
    )

    (
        hop_size_samples,
        win_size_samples,
        min_length_frames,
        min_interval_frames,
        max_sil_kept_frames,
    ) = _ms_to_frame_params(
        min_length_ms=min_length_ms,
        min_interval_ms=min_interval_ms,
        hop_size_ms=hop_size_ms,
        max_sil_kept_ms=max_sil_kept_ms,
    )

    linear_threshold = 10 ** (threshold_db / 20.0)

    rms_list = calculate_audio_rms(
        clean_audio,
        frame_length=win_size_samples,
        hop_length=hop_size_samples,
    )
    rms_1d = np.squeeze(rms_list)
    total_frames = int(rms_1d.shape[0])

    sil_tags = find_silence_boundaries(
        rms_list=rms_1d,
        threshold=linear_threshold,
        min_length=min_length_frames,
        min_interval=min_interval_frames,
        max_sil_kept=max_sil_kept_frames,
    )

    chunks = slice_waveform_by_boundaries(
        waveform=clean_audio,
        sil_tags=sil_tags,
        hop_size=hop_size_samples,
        total_frames=total_frames,
    )

    saved_paths: list[Path] = []
    for chunk in chunks:
        out_name = (
            f"{source_name}_{chunk.start_sample:010d}_{chunk.end_sample:010d}.wav"
        )
        out_path = output_dir / out_name
        sf.write(out_path, chunk.waveform, TARGET_SR, subtype="PCM_16")
        saved_paths.append(out_path)

    return saved_paths


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    temp_dir = project_root / "TEMP"

    parser = argparse.ArgumentParser(description="静音切片测试：输出 WAV 到 TEMP/")
    parser.add_argument(
        "input",
        nargs="?",
        default=str(temp_dir / "manbo.mp3"),
        help="输入音频路径（默认 TEMP/manbo.mp3）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(temp_dir),
        help="切片输出目录（默认 TEMP）",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    try:
        saved = slice_and_save(input_path, output_dir)
    except Exception as e:
        print(f"切片失败: {e}", file=sys.stderr)
        return 1

    print(f"源文件: {input_path}")
    print(f"切片数量: {len(saved)}")
    for p in saved:
        print(f"  {p.relative_to(project_root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
