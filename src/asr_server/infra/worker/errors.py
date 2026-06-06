"""Typed errors for worker subprocess dispatch."""

from __future__ import annotations

STDERR_TAIL_MAX_BYTES = 8192


class WorkerSpawnError(Exception):
    """Base error for worker spawn failures."""


class WorkerSpawnTimeout(WorkerSpawnError):
    def __init__(self, alias: str, timeout_sec: float) -> None:
        self.alias = alias
        self.timeout_sec = timeout_sec
        super().__init__(
            f"Worker '{alias}' timed out after {timeout_sec:g}s"
        )


class WorkerSpawnFailed(WorkerSpawnError):
    def __init__(self, alias: str, returncode: int, stderr_tail: str) -> None:
        self.alias = alias
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        detail = stderr_tail or f"exit code {returncode}"
        super().__init__(f"Worker '{alias}' failed: {detail}")


def tail_text(text: str, *, max_bytes: int = STDERR_TAIL_MAX_BYTES) -> str:
    """Return the last ``max_bytes`` of text for client-facing errors."""
    if not text:
        return ""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text.strip()
    return encoded[-max_bytes:].decode("utf-8", errors="replace").strip()
