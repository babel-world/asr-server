"""对 TEMP 中的切片 WAV 逐条转录，生成训练用 manifest CSV。

用法（项目根目录）::

    uv run python scripts/test_manifest.py
    uv run python scripts/test_manifest.py manbo
    uv run python scripts/test_manifest.py manbo --temp-dir TEMP

输出 ``TEMP/{base_stem}_manifest.csv``，表头::

    filename,speaker,language,text,probability

- ``filename``: 切片 WAV 文件名（与磁盘上的文件一致）
- ``speaker``: ``base_stem``（如 manbo）
- ``language`` / ``text`` / ``probability``: 与 ``TranscribeResponseBody`` 一致
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from asr_server.service.whisper_model import sync_transcribe_file

MANIFEST_FIELDS = ("filename", "speaker", "language", "text", "probability")


def collect_slice_wavs(temp_dir: Path, base_stem: str) -> list[Path]:
    """匹配 ``{base_stem}_*.wav``，排除 manifest 等非切片文件。"""
    pattern = f"{base_stem}_*.wav"
    paths = sorted(temp_dir.glob(pattern))
    return [p for p in paths if p.is_file() and "_manifest" not in p.stem]


def build_manifest(
    temp_dir: Path,
    base_stem: str,
    *,
    output_path: Path | None = None,
) -> tuple[Path, int]:
    wav_files = collect_slice_wavs(temp_dir, base_stem)
    if not wav_files:
        raise FileNotFoundError(
            f"未在 {temp_dir} 找到匹配 {base_stem}_*.wav 的切片文件"
        )

    out = output_path or (temp_dir / f"{base_stem}_manifest.csv")

    rows: list[dict[str, str | float]] = []
    total = len(wav_files)
    for i, wav_path in enumerate(wav_files, start=1):
        print(f"[{i}/{total}] 转录 {wav_path.name} ...", flush=True)
        result = sync_transcribe_file(wav_path)
        rows.append(
            {
                "filename": wav_path.name,
                "speaker": base_stem,
                "language": result.language,
                "text": result.transcribed_text,
                "probability": result.language_probability,
            }
        )

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return out, len(rows)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    default_temp = project_root / "TEMP"

    parser = argparse.ArgumentParser(
        description="为 TEMP 中的切片 WAV 生成转录 manifest CSV"
    )
    parser.add_argument(
        "base_stem",
        nargs="?",
        default="manbo",
        help="源音频主名（默认 manbo），匹配 {stem}_*.wav，输出 {stem}_manifest.csv",
    )
    parser.add_argument(
        "--temp-dir",
        default=str(default_temp),
        help="切片 WAV 所在目录（默认 TEMP）",
    )
    args = parser.parse_args()

    temp_dir = Path(args.temp_dir)
    if not temp_dir.is_absolute():
        temp_dir = project_root / temp_dir

    try:
        out_path, row_count = build_manifest(temp_dir, args.base_stem)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"转录失败: {e}", file=sys.stderr)
        return 1
    print(f"已写入 {out_path.relative_to(project_root)}，数据行数: {row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
