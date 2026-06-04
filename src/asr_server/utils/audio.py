import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly
from pydantic import BaseModel, ConfigDict
from typing import List, Tuple

"""音频切片流水线：模块级 I/O 契约一览。

+-----------------------------+------------------------------------------+------------------------------------------+----------------------------------+
| 名称                        | 输入 (Input)                             | 输出 (Output)                            | 职责                             |
+=============================+==========================================+==========================================+==================================+
| AudioChunk (BaseModel)      | —                                        | waveform: NDArray[np.float32]            | 数据容器：切片波形及其在原音频   |
|                             |                                          | start_sample: int                        | 中的绝对采样点区间               |
|                             |                                          | end_sample: int                          |                                  |
+-----------------------------+------------------------------------------+------------------------------------------+----------------------------------+
| normalize_audio             | raw_waveform: NDArray[np.float32]        | NDArray[np.float32]                      | 单声道 + 重采样（默认 32 kHz）   |
|                             |   (1D 或 2D)                             |   形状 ``(samples,)``                    |                                  |
|                             | source_sr: int（必选）                   |                                          |                                  |
|                             | target_sr: int（默认 32000）             |                                          |                                  |
+-----------------------------+------------------------------------------+------------------------------------------+----------------------------------+
| calculate_audio_rms         | audio_waveform: NDArray[np.float32]     | NDArray[np.float32]                      | 计算每帧 RMS 能量曲线            |
|                             | frame_length: int（默认 2048）           |   形状 ``(1, frames)`` 或                |                                  |
|                             | hop_length: int（默认 512）              |   ``(channels, 1, frames)``            |                                  |
+-----------------------------+------------------------------------------+------------------------------------------+----------------------------------+
| find_silence_boundaries     | rms_list: NDArray[np.float32]            | List[Tuple[int, int]]                    | 从能量曲线提取静音段起止帧索引   |
|                             | threshold: float                         |   ``(silence_start, silence_end)``       |                                  |
|                             | min_length / min_interval /              |                                          |                                  |
|                             | max_sil_kept: int（帧数）                |                                          |                                  |
+-----------------------------+------------------------------------------+------------------------------------------+----------------------------------+
| slice_waveform_by_boundaries| waveform: NDArray[np.float32] (1D)       | List[AudioChunk]                         | 按静音边界切分波形并映射采样点   |
|                             | sil_tags: List[Tuple[int, int]]          |                                          |                                  |
|                             | hop_size: int                            |                                          |                                  |
|                             | total_frames: int                        |                                          |                                  |
+-----------------------------+------------------------------------------+------------------------------------------+----------------------------------+
"""

class AudioChunk(BaseModel):
    """切片后的音频数据容器契约。

    Attributes:
        waveform: 实际切片出的波形片段。单声道，float32，形状 ``(samples,)``。
        start_sample: 该片段在原始音频流中的起始绝对采样点位置。
        end_sample: 该片段在原始音频流中的终止绝对采样点位置。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    waveform: NDArray[np.float32]
    start_sample: int
    end_sample: int


def normalize_audio(
    raw_waveform: NDArray[np.float32], 
    source_sr: int, 
    target_sr: int = 32000
) -> NDArray[np.float32]:
    """将波形转为单声道，并重采样到目标采样率。

    Args:
        raw_waveform: 输入波形，float32。``(samples,)`` 单声道；或二维
            ``(channels, samples)`` / ``(samples, channels)``（较短轴视为通道维并取均值）。
        source_sr: 输入采样率（Hz），必选；须为正整数，且与 ``raw_waveform`` 的实际采样率一致。
        target_sr: 目标采样率（Hz）。默认 32000。

    Returns:
        单声道波形，float32，形状 ``(samples,)``，采样率为 ``target_sr``。

    Raises:
        ValueError: ``raw_waveform`` 维度大于 2，或 ``source_sr`` / ``target_sr`` 不是正整数。
    """
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("source_sr 与 target_sr 必须为正整数（Hz）")

    if raw_waveform.ndim > 2:
        raise ValueError("仅支持单声道 (1D) 或双声道 (2D) 的音频数组")

    if raw_waveform.ndim == 2:
        if raw_waveform.shape[0] < raw_waveform.shape[1]:
            normalized_mono = raw_waveform.mean(axis=0)
        else:
            normalized_mono = raw_waveform.mean(axis=1)
    else:
        normalized_mono = raw_waveform

    if source_sr == target_sr:
        return normalized_mono.astype(np.float32)

    resampled_waveform = resample_poly(normalized_mono, target_sr, source_sr)
    
    return resampled_waveform.astype(np.float32)


def calculate_audio_rms(
    audio_waveform: NDArray[np.float32],
    frame_length: int = 2048,
    hop_length: int = 512,
) -> NDArray[np.float32]:
    """计算音频波形每一帧的均方根能量 (RMS)。逻辑来自 librosa.feature.rms。

    Args:
        audio_waveform: 音频波形，float32。形状 ``(samples,)`` 单声道，或
            ``(channels, samples)`` 多声道。
        frame_length: 分帧窗口长度（采样点数）。默认 2048；越大，局部平均窗口越长。
        hop_length: 帧移（采样点数）。默认 512；越小，输出帧的时间分辨率越高。

    Returns:
        每帧 RMS，float32。形状 ``(1, frames)`` 或 ``(channels, 1, frames)``。
    """
    pad_width = frame_length // 2
    pad_widths = [(0, 0)] * (audio_waveform.ndim - 1) + [(pad_width, pad_width)]
    
    padded_waveform = np.pad(audio_waveform, pad_width=pad_widths, mode="constant")

    time_axis = -1
    element_stride = padded_waveform.strides[time_axis]
    
    framed_shape = list(padded_waveform.shape)
    framed_shape[time_axis] -= (frame_length - 1)
    framed_shape = tuple(framed_shape) + (frame_length,)
    
    framed_strides = padded_waveform.strides + (element_stride,)
    
    framed_view = np.lib.stride_tricks.as_strided(
        padded_waveform, 
        shape=framed_shape, 
        strides=framed_strides
    )

    target_axis = time_axis - 1 if time_axis < 0 else time_axis + 1
    framed_view = np.moveaxis(framed_view, -1, target_axis)
    
    slices = [slice(None)] * framed_view.ndim
    slices[time_axis] = slice(0, None, hop_length)
    sampled_frames = framed_view[tuple(slices)]

    frame_power = np.mean(np.abs(sampled_frames) ** 2, axis=-2, keepdims=True)
    rms_energy = np.sqrt(frame_power)

    return rms_energy


def find_silence_boundaries(
    rms_list: NDArray[np.float32],
    threshold: float,
    min_length: int,
    min_interval: int,
    max_sil_kept: int
) -> List[Tuple[int, int]]:
    """扫描能量曲线提取静音边界的索引。

    Args:
        rms_list: 能量曲线数组，float32。形状应为 ``(frames,)``（若传入多维会自动 squeeze）。
        threshold: 静音判断阈值（线性标量，如 10 ** (dB / 20)）。
        min_length: 允许切片出的最小有效音频长度（帧数）。
        min_interval: 构成有效静音切割点的最小连续静音长度（帧数）。
        max_sil_kept: 切片前后保留的最大静音余量（帧数）。

    Returns:
        包含所有静音段起止帧索引的列表。格式为 ``[(silence_start_frame, silence_end_frame), ...]``。
    """
    if rms_list.ndim > 1:
        rms_list = np.squeeze(rms_list)

    sil_tags = []
    silence_start = None
    clip_start = 0

    for i, rms in enumerate(rms_list):
        if rms < threshold:
            if silence_start is None:
                silence_start = i
            continue

        if silence_start is None:
            continue

        is_leading_silence = silence_start == 0 and i > max_sil_kept
        need_slice_middle = i - silence_start >= min_interval and i - clip_start >= min_length
        
        if not is_leading_silence and not need_slice_middle:
            silence_start = None
            continue

        if i - silence_start <= max_sil_kept:
            pos = rms_list[silence_start : i + 1].argmin() + silence_start
            if silence_start == 0:
                sil_tags.append((0, pos))
            else:
                sil_tags.append((pos, pos))
            clip_start = pos
        elif i - silence_start <= max_sil_kept * 2:
            pos = rms_list[i - max_sil_kept : silence_start + max_sil_kept + 1].argmin()
            pos += i - max_sil_kept
            pos_l = rms_list[silence_start : silence_start + max_sil_kept + 1].argmin() + silence_start
            pos_r = rms_list[i - max_sil_kept : i + 1].argmin() + i - max_sil_kept
            if silence_start == 0:
                sil_tags.append((0, pos_r))
                clip_start = pos_r
            else:
                sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                clip_start = max(pos_r, pos)
        else:
            pos_l = rms_list[silence_start : silence_start + max_sil_kept + 1].argmin() + silence_start
            pos_r = rms_list[i - max_sil_kept : i + 1].argmin() + i - max_sil_kept
            if silence_start == 0:
                sil_tags.append((0, pos_r))
            else:
                sil_tags.append((pos_l, pos_r))
            clip_start = pos_r
            
        silence_start = None

    total_frames = rms_list.shape[0]
    if silence_start is not None and total_frames - silence_start >= min_interval:
        silence_end = min(total_frames, silence_start + max_sil_kept)
        pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
        sil_tags.append((pos, total_frames + 1))

    return sil_tags


def slice_waveform_by_boundaries(
    waveform: NDArray[np.float32],
    sil_tags: List[Tuple[int, int]],
    hop_size: int,
    total_frames: int
) -> List[AudioChunk]:
    """根据静音帧边界提取物理音频片段，并进行时域对齐。

    Args:
        waveform: 待切片的音频波形，float32。单声道，形状 ``(samples,)``。
        sil_tags: 静音边界列表。来自 ``find_silence_boundaries`` 的输出。
        hop_size: 帧移（每一帧对应的物理采样点数）。
        total_frames: 原音频的总帧数。

    Returns:
        一系列按时间顺序排列的 ``AudioChunk`` 对象。每个对象包含切片数组及原始采样位置。
    """
    num_samples = len(waveform)
    if len(sil_tags) == 0:
        return [AudioChunk(
            waveform=waveform,
            start_sample=0,
            end_sample=num_samples
        )]

    chunks = []
    
    # 1. 开头到第一个静音
    if sil_tags[0][0] > 0:
        start_s = 0
        end_s = min(num_samples, sil_tags[0][0] * hop_size)
        if start_s < end_s:
            chunks.append(AudioChunk(
                waveform=waveform[start_s:end_s],
                start_sample=start_s,
                end_sample=end_s
            ))

    # 2. 两个静音之间的语音段
    for i in range(len(sil_tags) - 1):
        voice_start_frame = sil_tags[i][1]
        voice_end_frame = sil_tags[i + 1][0]
        
        start_s = voice_start_frame * hop_size
        end_s = min(num_samples, voice_end_frame * hop_size)
        if start_s < end_s:
            chunks.append(AudioChunk(
                waveform=waveform[start_s:end_s],
                start_sample=start_s,
                end_sample=end_s
            ))

    # 3. 最后一个静音到结尾
    if sil_tags[-1][1] < total_frames:
        start_s = sil_tags[-1][1] * hop_size
        end_s = num_samples
        if start_s < end_s:
            chunks.append(AudioChunk(
                waveform=waveform[start_s:end_s],
                start_sample=start_s,
                end_sample=end_s
            ))

    return chunks