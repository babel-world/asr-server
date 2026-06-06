import asyncio
import io
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import UploadFile

from asr_server.utils.audio import (
    calculate_audio_rms,
    find_silence_boundaries,
    normalize_audio,
    slice_waveform_by_boundaries,
)
from asr_server.utils.manifest_zip import format_slice_zip_download_name

TARGET_SR = 32_000

# ZIP 内每个切片 WAV 的命名约定（供 /api/audio/slice 调用方解析）:
#   {base_name}_{chunk_index:04d}_{start:010d}-{end:010d}.wav
# - base_name: 上传文件主名（不含扩展名），如 manbo.mp3 -> manbo
# - chunk_index: 0-based，按时间顺序递增
# - start/end: 归一化后 32 kHz 单声道波形上的采样点索引，半开区间 [start, end)
# 示例: manbo_0000_0000000000-0000214720.wav


def format_slice_chunk_filename(
    base_name: str,
    chunk_index: int,
    start_sample: int,
    end_sample: int,
) -> str:
    """生成切片 WAV 文件名。

    格式: ``{base_name}_{chunk_index:04d}_{start:010d}-{end:010d}.wav``
    """
    return (
        f"{base_name}_{chunk_index:04d}_"
        f"{start_sample:010d}-{end_sample:010d}.wav"
    )


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


def sync_slice_and_zip(
    file_bytes: bytes,
    filename: str,
    *,
    threshold_db: float,
    min_length_ms: int,
    min_interval_ms: int,
    hop_size_ms: int,
    max_sil_kept_ms: int,
) -> tuple[Path, Path]:
    """切片并打包为 ZIP。返回 ``(zip_path, session_dir)``，由调用方负责清理 ``session_dir``。

    ZIP 内每个 WAV 的文件名见模块顶部约定及 ``format_slice_chunk_filename``。
    """
    session_dir = Path(tempfile.mkdtemp(prefix="asr_slice_"))
    chunks_dir = session_dir / "chunks"
    chunks_dir.mkdir()

    base_name = Path(filename).stem or "audio"

    raw_waveform, source_sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
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

    for chunk_index, chunk in enumerate(chunks):
        out_name = format_slice_chunk_filename(
            base_name, chunk_index, chunk.start_sample, chunk.end_sample
        )
        sf.write(
            chunks_dir / out_name,
            chunk.waveform,
            TARGET_SR,
            subtype="PCM_16",
        )

    zip_base = session_dir / "slices"
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=chunks_dir))
    return Path(zip_path), session_dir


async def slice_upload_to_zip(
    file: UploadFile,
    *,
    threshold_db: float,
    min_length_ms: int,
    min_interval_ms: int,
    hop_size_ms: int,
    max_sil_kept_ms: int,
) -> tuple[Path, Path, str]:
    """异步切片并打包。返回 ``(zip_path, session_dir, download_filename)``。"""
    file_bytes = await file.read()
    filename = file.filename or "audio"
    zip_path, session_dir = await asyncio.to_thread(
        sync_slice_and_zip,
        file_bytes,
        filename,
        threshold_db=threshold_db,
        min_length_ms=min_length_ms,
        min_interval_ms=min_interval_ms,
        hop_size_ms=hop_size_ms,
        max_sil_kept_ms=max_sil_kept_ms,
    )
    download_name = format_slice_zip_download_name(filename)
    return zip_path, session_dir, download_name
