"""Hello-world entry point for the chinese-hubert-base worker."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TencentGameMate/chinese-hubert-base worker (scaffold).",
    )
    parser.parse_args(argv)
    print("Hello from chinese-hubert-base worker")


if __name__ == "__main__":
    main()
