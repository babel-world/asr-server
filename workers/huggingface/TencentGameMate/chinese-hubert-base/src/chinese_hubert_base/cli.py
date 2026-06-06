"""CLI entry point for the chinese-hubert-base worker."""

from __future__ import annotations

import argparse
from pathlib import Path

from chinese_hubert_base import download, extract


def _cmd_download(_args: argparse.Namespace) -> int:
    model_path, source = download.run_download()
    print(f"model_path={model_path}")
    print(f"source={source}")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    output_path = extract.run_extract(args.input, args.output)
    print(f"input={args.input}")
    print(f"output={output_path}")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TencentGameMate/chinese-hubert-base worker.",
    )
    subparsers = parser.add_subparsers(dest="command")

    download_parser = subparsers.add_parser(
        "download",
        help="Download or resolve the model into .models",
    )
    download_parser.set_defaults(handler=_cmd_download)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract HuBERT features from a waveform .npy file (test/CLI shell)",
    )
    extract_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="PATH",
        help="Input waveform .npy (float32, 16 kHz mono, shape (T,) or (1, T))",
    )
    extract_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Output feature .npy path (last_hidden_state, shape (1, T', 768))",
    )
    extract_parser.set_defaults(handler=_cmd_extract)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(0)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
