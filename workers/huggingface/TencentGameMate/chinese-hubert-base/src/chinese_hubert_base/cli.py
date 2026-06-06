"""CLI entry point for the chinese-hubert-base worker."""

from __future__ import annotations

import argparse

from chinese_hubert_base import download


def _cmd_download(_args: argparse.Namespace) -> int:
    model_path, source = download.run_download()
    print(f"model_path={model_path}")
    print(f"source={source}")
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

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(0)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
