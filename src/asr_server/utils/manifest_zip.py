"""切片 ZIP 与 WAV 文件名的解析、解压与校验（供 /api/manifest 使用）。"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# 与 format_slice_chunk_filename 一致: {base_name}_{chunk_index:04d}_{start:010d}-{end:010d}.wav
SLICE_CHUNK_FILENAME_RE = re.compile(
    r"^(.+)_(\d{4})_(\d{10})-(\d{10})\.wav$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SliceChunkFilenameParts:
    """从切片 WAV 文件名解析出的字段（数字为 int，逻辑上与文件名中的前导零等价）。"""

    filename: str
    base_name: str
    chunk_index: int
    start_sample: int
    end_sample: int


def parse_slice_chunk_filename(filename: str) -> SliceChunkFilenameParts:
    """解析 ``{base_name}_{chunk_index:04d}_{start:010d}-{end:010d}.wav``。

    Raises:
        ValueError: 文件名不符合约定。
    """
    name = Path(filename).name
    match = SLICE_CHUNK_FILENAME_RE.match(name)
    if not match:
        raise ValueError(
            f"文件名不符合切片约定: {name!r}，"
            "期望 {base_name}_{chunk_index:04d}_{start:010d}-{end:010d}.wav"
        )
    base_name, idx, start, end = match.groups()
    return SliceChunkFilenameParts(
        filename=name,
        base_name=base_name,
        chunk_index=int(idx, 10),
        start_sample=int(start, 10),
        end_sample=int(end, 10),
    )


def format_slice_zip_download_name(source_filename: str) -> str:
    """切片 ZIP 下载名: ``{stem}_slices.zip``（不含源文件扩展名中的点）。"""
    stem = Path(source_filename).stem or "audio"
    return f"{stem}_slices.zip"


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """将 ZIP 解压到 ``dest_dir``，拒绝路径穿越。"""
    dest_root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"ZIP 含非法路径: {info.filename}")
            target = (dest_root / Path(*member.parts)).resolve()
            if not str(target).startswith(str(dest_root)):
                raise ValueError(f"ZIP 路径穿越: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())


def collect_slice_wavs_from_dir(extract_dir: Path) -> list[SliceChunkFilenameParts]:
    """扫描目录内所有 ``.wav``，校验命名且 ``base_name`` 唯一，按 ``chunk_index`` 排序。

    Returns:
        解析结果列表（仅元数据，不含磁盘路径）。

    Raises:
        ValueError: 无 wav、命名非法或 base_name 不一致。
    """
    wav_paths = sorted(extract_dir.rglob("*.wav"))
    if not wav_paths:
        raise ValueError("ZIP 中未找到任何 .wav 文件")

    entries: list[SliceChunkFilenameParts] = []
    for path in wav_paths:
        entries.append(parse_slice_chunk_filename(path.name))

    base_names = {e.base_name for e in entries}
    if len(base_names) != 1:
        raise ValueError(
            f"ZIP 内 base_name 不一致: {sorted(base_names)}，要求全部相同"
        )

    entries.sort(key=lambda e: e.chunk_index)
    indices = [e.chunk_index for e in entries]
    if len(indices) != len(set(indices)):
        raise ValueError("ZIP 内存在重复的 chunk_index")

    return entries


def resolve_wav_path(extract_dir: Path, entry: SliceChunkFilenameParts) -> Path:
    """根据文件名在解压目录中定位 WAV（忽略 ZIP 内子目录深度，按 basename 匹配）。"""
    matches = [p for p in extract_dir.rglob("*.wav") if p.name == entry.filename]
    if len(matches) != 1:
        raise ValueError(f"无法唯一定位 WAV: {entry.filename}")
    return matches[0]
