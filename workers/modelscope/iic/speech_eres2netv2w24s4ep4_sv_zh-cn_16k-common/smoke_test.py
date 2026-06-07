"""SV Worker smoke test: in-memory + disk extract paths.

Usage (from worker directory):
  uv run python smoke_test.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

WORKER_DIR = Path(__file__).resolve().parent
ALIAS = "speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common"


def _pick_test_waveform() -> np.ndarray:
    """Return a short 16 kHz mono float32 waveform for smoke testing."""
    examples = (
        WORKER_DIR
        / ".models"
        / "speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common"
        / "examples"
    )
    for name in ("speaker1_a_cn_16k.wav", "speaker1_b_cn_16k.wav"):
        wav_path = examples / name
        if wav_path.is_file():
            try:
                import librosa
            except ImportError:
                break
            audio, _ = librosa.load(wav_path, sr=16000, mono=True)
            peak = float(np.abs(audio).max())
            if peak > 0:
                audio = audio / peak * 0.95
            return audio.astype(np.float32, copy=False)

    local_npy = WORKER_DIR / ".local" / "sv_input.npy"
    if local_npy.is_file():
        return np.load(local_npy).astype(np.float32, copy=False)

    # Fallback: synthetic tone @ 16 kHz
    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(16000 * duration_sec), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return tone


def _assert_output(out: np.ndarray) -> None:
    assert out.shape == (1, 20480), f"unexpected shape: {out.shape}"
    assert out.dtype == np.float32, f"unexpected dtype: {out.dtype}"


def test_memory_and_disk() -> None:
    from speech_eres2netv2w24s4ep4_sv_zh_cn_16k_common.extract import (
        extract_sv_single,
        run_extract,
    )

    waveform = _pick_test_waveform()
    mem_out = extract_sv_single(waveform)
    _assert_output(mem_out)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_npy = tmp_dir / "sv_input.npy"
        output_npy = tmp_dir / "sv_output.npy"
        np.save(input_npy, waveform)
        run_extract(input_npy, output_npy)
        disk_out = np.load(output_npy)
        _assert_output(disk_out)

    print("[OK] memory + disk extract")


def test_cli_extract() -> None:
    waveform = _pick_test_waveform()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_npy = tmp_dir / "sv_input.npy"
        output_npy = tmp_dir / "sv_output.npy"
        np.save(input_npy, waveform)
        result = subprocess.run(
            [
                "uv",
                "run",
                ALIAS,
                "extract",
                "--input",
                str(input_npy),
                "--output",
                str(output_npy),
            ],
            cwd=WORKER_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"CLI extract failed:\n{result.stderr}")
        out = np.load(output_npy)
        _assert_output(out)
    print("[OK] CLI extract")


def test_serve_protocol() -> None:
    waveform = _pick_test_waveform()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_npy = tmp_dir / "sv_input.npy"
        output_npy = tmp_dir / "sv_output.npy"
        np.save(input_npy, waveform)

        proc = subprocess.Popen(
            ["uv", "run", ALIAS, "serve"],
            cwd=WORKER_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None

        ready_line = proc.stdout.readline().strip()
        ready = json.loads(ready_line)
        assert ready.get("ok") is True and ready.get("event") == "ready"

        request = json.dumps(
            {"cmd": "extract", "input": str(input_npy), "output": str(output_npy)}
        )
        proc.stdin.write(request + "\n")
        proc.stdin.flush()
        response_line = proc.stdout.readline().strip()
        response = json.loads(response_line)
        assert response.get("ok") is True

        proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
        proc.stdin.flush()
        shutdown_line = proc.stdout.readline().strip()
        shutdown = json.loads(shutdown_line)
        assert shutdown.get("ok") is True and shutdown.get("event") == "shutdown"

        proc.wait(timeout=120)
        out = np.load(output_npy)
        _assert_output(out)
    print("[OK] serve protocol")


def test_run_worker_dispatch() -> None:
    asr_server_root = WORKER_DIR.parent.parent.parent.parent
    run_worker = asr_server_root / "workers" / "run_worker.py"
    waveform = _pick_test_waveform()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_npy = tmp_dir / "sv_input.npy"
        output_npy = tmp_dir / "sv_output.npy"
        np.save(input_npy, waveform)
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(run_worker),
                ALIAS,
                "--",
                "extract",
                "--input",
                str(input_npy),
                "--output",
                str(output_npy),
            ],
            cwd=asr_server_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"run_worker dispatch failed (code={result.returncode}):\n"
                f"stdout={result.stdout}\nstderr={result.stderr}"
            )
        out = np.load(output_npy)
        _assert_output(out)
    print("[OK] run_worker.py dispatch")


def main() -> None:
    print(f"Worker dir: {WORKER_DIR}")
    test_memory_and_disk()
    test_cli_extract()
    test_serve_protocol()
    test_run_worker_dispatch()
    print("[OK] SV Worker smoke tests passed")


if __name__ == "__main__":
    main()
