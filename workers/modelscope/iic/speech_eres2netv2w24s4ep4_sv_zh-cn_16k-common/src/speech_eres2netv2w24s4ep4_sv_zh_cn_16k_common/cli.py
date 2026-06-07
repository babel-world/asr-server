"""CLI entry point for the speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common worker."""

from __future__ import annotations

import argparse
from pathlib import Path

from speech_eres2netv2w24s4ep4_sv_zh_cn_16k_common import download, extract, serve


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


def _cmd_serve(_args: argparse.Namespace) -> int:
    return serve.run_serve()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common SV worker.",
    )
    subparsers = parser.add_subparsers(dest="command")

    download_parser = subparsers.add_parser(
        "download",
        help="Download or resolve the model into .models",
    )
    download_parser.set_defaults(handler=_cmd_download)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract SV features from sv_input.npy (16 kHz float32 mono)",
    )
    extract_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="PATH",
        help="Input waveform .npy, shape (T,) or (1, T)",
    )
    extract_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Output sv_output.npy, shape (1, 20480)",
    )
    extract_parser.set_defaults(handler=_cmd_extract)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run long-lived extract server (JSON lines on stdin; for asr-server batch)",
    )
    serve_parser.set_defaults(handler=_cmd_serve)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(0)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
