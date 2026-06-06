"""Invoke registered workers via workers/run_worker.py (subprocess-safe spawn)."""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache

from asr_server.infra.worker.config import get_worker_timeout_sec
from asr_server.infra.worker.errors import (
    WorkerSpawnFailed,
    WorkerSpawnTimeout,
    tail_text,
)
from asr_server.utils.whisper_assets import get_repo_root

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpawnResult:
    alias: str
    returncode: int
    duration_ms: float


@lru_cache(maxsize=1)
def _load_run_worker_module():
    module_path = get_repo_root() / "workers" / "run_worker.py"
    spec = importlib.util.spec_from_file_location("run_worker", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load worker dispatcher: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spawn_worker(alias: str, worker_args: list[str]) -> SpawnResult:
    """Run ``uv run <cli> ...`` for a registered worker; raise on timeout/failure."""
    timeout_sec = get_worker_timeout_sec(alias)
    run_worker = _load_run_worker_module()
    started = time.perf_counter()

    try:
        completed = run_worker.spawn_worker(
            alias,
            worker_args,
            timeout_sec=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        logger.error(
            "worker spawn timeout alias=%s timeout_sec=%s duration_ms=%.1f",
            alias,
            timeout_sec,
            duration_ms,
        )
        raise WorkerSpawnTimeout(alias, timeout_sec) from exc

    duration_ms = (time.perf_counter() - started) * 1000

    if completed.returncode != 0:
        stderr_full = completed.stderr or completed.stdout or ""
        stderr_tail = tail_text(stderr_full)
        logger.error(
            "worker spawn failed alias=%s returncode=%s duration_ms=%.1f stderr=%s",
            alias,
            completed.returncode,
            duration_ms,
            stderr_full,
        )
        raise WorkerSpawnFailed(alias, completed.returncode, stderr_tail)

    logger.info(
        "worker spawn ok alias=%s returncode=0 duration_ms=%.1f timeout_sec=%s",
        alias,
        duration_ms,
        timeout_sec,
    )
    return SpawnResult(
        alias=alias,
        returncode=completed.returncode,
        duration_ms=duration_ms,
    )
