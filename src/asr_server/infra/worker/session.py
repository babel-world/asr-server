"""Persistent worker subprocess sessions (model loaded once until stop)."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asr_server.infra.worker.config import get_worker_timeout_sec
from asr_server.infra.worker.dispatch import _load_run_worker_module
from asr_server.infra.worker.errors import WorkerSpawnFailed, WorkerSpawnTimeout, tail_text

logger = logging.getLogger(__name__)

SESSION_START_TIMEOUT_SEC = 300.0
SESSION_STOP_TIMEOUT_SEC = 30.0


@dataclass(frozen=True)
class SessionStartResult:
    newly_started: bool
    loaded: bool


def _read_json_response(stream, timeout_sec: float, *, alias: str) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_sec
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"timed out after {timeout_sec:g}s waiting for worker response")
        try:
            line = _readline_with_timeout(stream, remaining)
        except TimeoutError:
            raise TimeoutError(
                f"timed out after {timeout_sec:g}s waiting for worker response"
            ) from None
        if not line:
            raise WorkerSpawnFailed(alias, 1, "worker closed stdout")
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning(
                "worker non-json stdout alias=%s line=%r",
                alias,
                stripped[:200],
            )
            continue
        return payload


def _readline_with_timeout(stream, timeout_sec: float) -> str:
    box: list[str] = []

    def _read() -> None:
        line = stream.readline()
        if line:
            box.append(line)

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        raise TimeoutError(f"timed out after {timeout_sec:g}s waiting for worker response")
    if not box:
        return ""
    return box[0]


class _StderrDrainer:
    """Background reader so worker stderr cannot fill the pipe and deadlock."""

    def __init__(self, proc: subprocess.Popen[str], alias: str) -> None:
        self._alias = alias
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            args=(proc,),
            name=f"worker-stderr-{alias}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def tail(self, max_bytes: int = 8192) -> str:
        with self._lock:
            return tail_text("".join(self._lines), max_bytes=max_bytes)

    def _run(self, proc: subprocess.Popen[str]) -> None:
        stderr = proc.stderr
        if stderr is None:
            return
        try:
            for line in stderr:
                with self._lock:
                    self._lines.append(line)
                stripped = line.rstrip("\n")
                if stripped:
                    logger.debug(
                        "worker stderr alias=%s line=%s",
                        self._alias,
                        stripped[:500],
                    )
        except (OSError, ValueError):
            logger.debug("worker stderr drainer stopped alias=%s", self._alias)


class PersistentWorkerSession:
    """One long-lived ``uv run <worker> serve`` process per alias."""

    def __init__(self, alias: str) -> None:
        self.alias = alias
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_drainer: _StderrDrainer | None = None

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(self) -> SessionStartResult:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return SessionStartResult(newly_started=False, loaded=True)
            self._start_locked()
            return SessionStartResult(newly_started=True, loaded=True)

    def stop(self) -> bool:
        with self._lock:
            if self._proc is None:
                return False
            if self._proc.poll() is not None:
                self._proc = None
                self._stderr_drainer = None
                return True
            self._shutdown_locked()
            return True

    def ensure_started(self) -> SessionStartResult:
        return self.start()

    def extract_npy(self, input_path: Path, output_path: Path) -> None:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start_locked()
            self._request_locked(
                {
                    "cmd": "extract",
                    "input": str(input_path),
                    "output": str(output_path),
                },
                timeout_sec=get_worker_timeout_sec(self.alias),
            )

    def _start_locked(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        run_worker = _load_run_worker_module()
        command, worker_dir = run_worker.build_worker_command(self.alias, ["serve"])
        started = time.perf_counter()
        try:
            proc = subprocess.Popen(
                command,
                cwd=worker_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to start worker '{self.alias}': {exc}") from exc

        self._stderr_drainer = _StderrDrainer(proc, self.alias)
        self._stderr_drainer.start()

        try:
            ready = _read_json_response(
                proc.stdout,
                SESSION_START_TIMEOUT_SEC,
                alias=self.alias,
            )
        except TimeoutError as exc:
            self._clear_process(proc)
            raise WorkerSpawnTimeout(self.alias, SESSION_START_TIMEOUT_SEC) from exc

        if proc.poll() is not None:
            stderr = self._stderr_tail(proc)
            self._clear_process(proc)
            raise WorkerSpawnFailed(
                self.alias,
                proc.returncode or 1,
                stderr or "worker exited during startup",
            )

        if not ready.get("ok"):
            stderr = self._stderr_tail(proc)
            self._clear_process(proc)
            raise WorkerSpawnFailed(
                self.alias,
                1,
                stderr or str(ready.get("error") or "worker failed to become ready"),
            )

        self._proc = proc
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "worker session started alias=%s duration_ms=%.1f",
            self.alias,
            duration_ms,
        )

    def _shutdown_locked(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        self._stderr_drainer = None

        if proc.poll() is not None:
            return

        try:
            self._write_request(proc, {"cmd": "shutdown"})
            _readline_with_timeout(proc.stdout, SESSION_STOP_TIMEOUT_SEC)
        except (TimeoutError, OSError, BrokenPipeError):
            logger.warning("worker session shutdown timed out alias=%s", self.alias)
        finally:
            self._terminate_process(proc)
            logger.info("worker session stopped alias=%s", self.alias)

    def _request_locked(self, payload: dict[str, Any], *, timeout_sec: float) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError(f"Worker session '{self.alias}' is not running")

        if proc.poll() is not None:
            stderr = self._stderr_tail(proc)
            self._clear_process(proc)
            raise WorkerSpawnFailed(
                self.alias,
                proc.returncode or 1,
                stderr or "worker exited before request",
            )

        try:
            self._write_request(proc, payload)
            response = _read_json_response(
                proc.stdout,
                timeout_sec,
                alias=self.alias,
            )
        except TimeoutError as exc:
            self._clear_process(proc)
            raise WorkerSpawnTimeout(self.alias, timeout_sec) from exc
        except OSError as exc:
            self._clear_process(proc)
            raise WorkerSpawnFailed(self.alias, 1, str(exc)) from exc

        if proc.poll() is not None:
            stderr = self._stderr_tail(proc)
            self._clear_process(proc)
            raise WorkerSpawnFailed(
                self.alias,
                proc.returncode or 1,
                stderr or "worker exited during request",
            )

        if not response.get("ok"):
            raise WorkerSpawnFailed(
                self.alias,
                1,
                str(response.get("error") or "worker request failed"),
            )

    def _stderr_tail(self, proc: subprocess.Popen[str]) -> str:
        if self._stderr_drainer is not None:
            return self._stderr_drainer.tail()
        return tail_text(proc.stderr.read() if proc.stderr else "")

    def _clear_process(self, proc: subprocess.Popen[str]) -> None:
        self._terminate_process(proc)
        self._proc = None
        self._stderr_drainer = None

    @staticmethod
    def _write_request(proc: subprocess.Popen[str], payload: dict[str, Any]) -> None:
        if proc.stdin is None:
            raise RuntimeError("worker stdin is closed")
        proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proc.stdin.flush()

    @staticmethod
    def _terminate_process(proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


_sessions_lock = threading.Lock()
_sessions: dict[str, PersistentWorkerSession] = {}


def get_worker_session(alias: str) -> PersistentWorkerSession:
    with _sessions_lock:
        session = _sessions.get(alias)
        if session is None:
            session = PersistentWorkerSession(alias)
            _sessions[alias] = session
        return session
