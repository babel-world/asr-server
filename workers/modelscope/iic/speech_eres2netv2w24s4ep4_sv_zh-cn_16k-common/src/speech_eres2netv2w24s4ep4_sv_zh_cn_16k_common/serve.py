"""Long-running serve loop: load model once, handle JSON-line requests on stdin."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from speech_eres2netv2w24s4ep4_sv_zh_cn_16k_common import extract


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle_request(payload: dict[str, Any]) -> dict[str, Any]:
    cmd = payload.get("cmd")
    if cmd == "ping":
        extract.get_runtime()
        return {"ok": True, "loaded": True}
    if cmd == "extract":
        input_path = Path(payload["input"])
        output_path = Path(payload["output"])
        extract.run_extract(input_path, output_path)
        return {"ok": True}
    if cmd == "shutdown":
        return {"ok": True, "event": "shutdown"}
    return {"ok": False, "error": f"unknown cmd: {cmd!r}"}


def run_serve() -> int:
    """Load ERes2NetV2 once, then process one JSON object per stdin line."""
    try:
        extract.get_runtime()
    except Exception as exc:
        _write_response({"ok": False, "error": str(exc), "event": "ready"})
        return 1

    _write_response({"ok": True, "event": "ready"})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_response({"ok": False, "error": f"invalid json: {exc}"})
            continue

        if payload.get("cmd") == "shutdown":
            _write_response({"ok": True, "event": "shutdown"})
            return 0

        try:
            response = _handle_request(payload)
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        _write_response(response)

    return 0
