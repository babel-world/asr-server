"""验证 POST /api/transcribe/start 与 /api/transcribe/stop 的行为与时延差异。

脚本会自动管理 asr-server 生命周期（含 19031 端口占用清理）::

    uv sync --extra dev
    uv run python scripts/test_start_stop.py
    uv run python scripts/test_start_stop.py --rounds 3
    uv run python scripts/test_start_stop.py --wav .local/TEMP/manbo_0035_0006602240-0006684160.wav

报告输出目录：.local/TEMP/reports/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "http://127.0.0.1:19031"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 19031
DEFAULT_SAMPLE_DIR = REPO_ROOT / ".local" / "TEMP"
DEFAULT_REPORTS_DIR = REPO_ROOT / ".local" / "TEMP" / "reports"
DEFAULT_WAV = DEFAULT_SAMPLE_DIR / "manbo_0035_0006602240-0006684160.wav"
DEFAULT_ROUNDS = 5
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_STOP_SLOWER_RATIO = 0.8
BASELINE_REPORT_JSON = DEFAULT_REPORTS_DIR / "start_stop_check_20260604_225553.json"
BASELINE_COMMIT = "6ac704ce58e1fc4ddff77e2fdbd474f209aeaa2a"
SERVER_STARTUP_TIMEOUT_S = 120.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check /api/transcribe/start and /api/transcribe/stop with latency comparison.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Server base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--wav",
        type=Path,
        default=None,
        help="WAV file for transcribe requests (default: random from sample-dir)",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=DEFAULT_SAMPLE_DIR,
        help="Directory to search when --wav is not provided explicitly",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory for JSON and Markdown reports",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"Latency comparison rounds (default: {DEFAULT_ROUNDS})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--stop-slower-ratio",
        type=float,
        default=DEFAULT_STOP_SLOWER_RATIO,
        help=(
            "Minimum fraction of rounds where stop->transcribe is slower than "
            f"start->transcribe (default: {DEFAULT_STOP_SLOWER_RATIO})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for WAV selection when --wav is omitted",
    )
    parser.add_argument(
        "--no-manage-server",
        action="store_true",
        help="Do not start/stop asr-server automatically (assume already running)",
    )
    return parser.parse_args()


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_wav(args: argparse.Namespace) -> Path:
    if args.wav is not None:
        wav = args.wav if args.wav.is_absolute() else REPO_ROOT / args.wav
        if wav.exists():
            return wav.resolve()
        print(f"WAV not found: {wav}", file=sys.stderr)
        raise SystemExit(1)

    sample_dir = args.sample_dir if args.sample_dir.is_absolute() else REPO_ROOT / args.sample_dir
    candidates = sorted(sample_dir.glob("*.wav"))
    if not candidates:
        print(f"No .wav files found in {sample_dir}", file=sys.stderr)
        raise SystemExit(1)
    rng = random.Random(args.seed)
    return rng.choice(candidates).resolve()


def _pids_on_port(host: str, port: int) -> list[int]:
    if sys.platform == "win32":
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            check=False,
        )
        needle = f"{host}:{port}"
        pids: set[int] = set()
        for line in result.stdout.splitlines():
            if needle not in line or "LISTENING" not in line:
                continue
            parts = line.split()
            if parts:
                try:
                    pids.add(int(parts[-1]))
                except ValueError:
                    continue
        return sorted(pids)

    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _process_name(pid: int) -> str | None:
    if sys.platform == "win32":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        line = result.stdout.strip()
        if not line or "No tasks" in line:
            return None
        return line.split(",")[0].strip('"').lower()

    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "comm="],
        capture_output=True,
        text=True,
        check=False,
    )
    name = result.stdout.strip().lower()
    return name or None


def _is_python_process(pid: int) -> bool:
    name = _process_name(pid) or ""
    return "python" in name


def _kill_pid(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False)
        return
    subprocess.run(["kill", "-9", str(pid)], check=False)


def _ensure_port_available(host: str, port: int) -> dict[str, Any]:
    """Free port if occupied by python; abort otherwise."""
    record: dict[str, Any] = {
        "host": host,
        "port": port,
        "initial_pids": [],
        "killed_pids": [],
        "blocked_pids": [],
        "action": "none",
    }
    pids = _pids_on_port(host, port)
    record["initial_pids"] = pids
    if not pids:
        return record

    for pid in pids:
        if _is_python_process(pid):
            _kill_pid(pid)
            record["killed_pids"].append(pid)
        else:
            record["blocked_pids"].append(
                {"pid": pid, "process_name": _process_name(pid)}
            )

    if record["blocked_pids"]:
        record["action"] = "blocked_non_python"
        return record

    record["action"] = "killed_python" if record["killed_pids"] else "none"
    time.sleep(1.0)
    return record


def _wait_for_server(base_url: str, timeout_s: float) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            resp = httpx.get(f"{base_url}/docs", timeout=2.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@contextmanager
def _managed_server(base_url: str, manage_server: bool) -> Iterator[dict[str, Any]]:
    lifecycle: dict[str, Any] = {
        "managed": manage_server,
        "port_guard": None,
        "server_pid": None,
        "startup_ok": None,
    }
    proc: subprocess.Popen[str] | None = None

    parsed = urlparse(base_url)
    host = parsed.hostname or DEFAULT_HOST
    port = parsed.port or DEFAULT_PORT

    if not manage_server:
        yield lifecycle
        return

    lifecycle["port_guard"] = _ensure_port_available(host, port)
    if lifecycle["port_guard"]["action"] == "blocked_non_python":
        blocked = lifecycle["port_guard"]["blocked_pids"]
        raise RuntimeError(
            f"Port {host}:{port} is occupied by non-python process: {blocked}"
        )

    proc = subprocess.Popen(
        ["uv", "run", "asr-server"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lifecycle["server_pid"] = proc.pid
    lifecycle["startup_ok"] = _wait_for_server(base_url, SERVER_STARTUP_TIMEOUT_S)
    if not lifecycle["startup_ok"]:
        proc.terminate()
        raise RuntimeError(f"Server failed to start within {SERVER_STARTUP_TIMEOUT_S}s")

    try:
        yield lifecycle
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def _collect_env_snapshot() -> dict[str, str | None]:
    keys = (
        "WHISPER_MODEL",
        "WHISPER_DEVICE",
        "WHISPER_COMPUTE_TYPE",
        "WHISPER_MODELS_DIR",
        "ASR_REPO_ROOT",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
    )
    return {k: os.getenv(k) for k in keys}


def _collect_model_layout() -> dict[str, Any]:
    model_name = os.getenv("WHISPER_MODEL", "base")
    try:
        from asr_server.utils.whisper_assets import (
            ensure_whisper_model_path,
            find_hf_snapshot,
            get_local_models_dir,
            is_ct2_model_dir,
            local_model_dir,
        )

        local_path = local_model_dir(model_name)
        hf_path = find_hf_snapshot(model_name)
        resolved_path, source = ensure_whisper_model_path(model_name)
        return {
            "whisper_model": model_name,
            "local_models_dir": str(get_local_models_dir()),
            "local_model_path": str(local_path),
            "local_model_valid": is_ct2_model_dir(local_path),
            "hf_snapshot_path": str(hf_path) if hf_path else None,
            "resolved_path": str(resolved_path),
            "resolved_source": source,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _git_snapshot() -> dict[str, Any]:
    def _run(cmd: list[str]) -> str:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return (result.stdout or result.stderr).strip()

    status_short = _run(["git", "status", "--short"])
    diff_names = _run(["git", "diff", "--name-only", BASELINE_COMMIT])
    untracked = [
        line[3:].strip()
        for line in status_short.splitlines()
        if line.startswith("??")
    ]
    changed = [line.strip() for line in diff_names.splitlines() if line.strip()]
    changed.extend(untracked)
    return {
        "head": _run(["git", "rev-parse", "HEAD"]),
        "baseline_commit": BASELINE_COMMIT,
        "status_short": status_short,
        "diff_names": diff_names,
        "untracked_files": untracked,
        "all_changed_files": sorted(set(changed)),
    }


def _post_json(client: httpx.Client, url: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        resp = client.post(url)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:500]}
        return {
            "url": url,
            "status_code": resp.status_code,
            "elapsed_ms": elapsed_ms,
            "body": body,
            "error": None,
        }
    except Exception as exc:
        return {
            "url": url,
            "status_code": None,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "body": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _post_transcribe(client: httpx.Client, transcribe_url: str, wav: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        with wav.open("rb") as f:
            resp = client.post(
                transcribe_url,
                files={"file": (wav.name, f, "audio/wav")},
            )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:500]}
        return {
            "url": transcribe_url,
            "status_code": resp.status_code,
            "elapsed_ms": elapsed_ms,
            "body": body,
            "summary": {
                "language": body.get("language"),
                "languageProbability": body.get("languageProbability"),
                "textLen": len(body.get("transcribedText", "")),
            },
            "error": None,
        }
    except Exception as exc:
        return {
            "url": transcribe_url,
            "status_code": None,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "body": None,
            "summary": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _ok_step(step: dict[str, Any]) -> bool:
    return not step.get("error") and step.get("status_code") == 200


def _run_basic_checks(
    client: httpx.Client,
    transcribe_url: str,
    start_url: str,
    stop_url: str,
    wav: Path,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    sequence: list[tuple[str, str | None]] = [
        ("basic_stop_cold", stop_url),
        ("basic_transcribe_implicit", None),
        ("basic_stop_after_transcribe", stop_url),
        ("basic_start", start_url),
        ("basic_start_idempotent", start_url),
        ("basic_transcribe_after_start", None),
    ]
    for name, url in sequence:
        step = (
            _post_transcribe(client, transcribe_url, wav)
            if url is None
            else _post_json(client, url)
        )
        steps.append({"name": name, **step})
    return steps


def _run_latency_round(
    client: httpx.Client,
    transcribe_url: str,
    start_url: str,
    stop_url: str,
    wav: Path,
    round_idx: int,
) -> dict[str, Any]:
    round_data: dict[str, Any] = {"round": round_idx, "scenario_a": {}, "scenario_b": {}}

    stop_a = _post_json(client, stop_url)
    transcribe_after_stop = _post_transcribe(client, transcribe_url, wav)
    round_data["scenario_a"]["stop"] = stop_a
    round_data["scenario_a"]["transcribe"] = transcribe_after_stop
    round_data["latency_after_stop_ms"] = transcribe_after_stop["elapsed_ms"]

    stop_b = _post_json(client, stop_url)
    start_b = _post_json(client, start_url)
    transcribe_after_start = _post_transcribe(client, transcribe_url, wav)
    round_data["scenario_b"]["stop"] = stop_b
    round_data["scenario_b"]["start"] = start_b
    round_data["scenario_b"]["transcribe"] = transcribe_after_start
    round_data["latency_after_start_ms"] = transcribe_after_start["elapsed_ms"]

    stop_ms = round_data["latency_after_stop_ms"]
    start_ms = round_data["latency_after_start_ms"]
    round_data["ratio"] = round(stop_ms / start_ms, 3) if start_ms else None
    round_data["stop_slower"] = stop_ms > start_ms if start_ms else False
    return round_data


def _p95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    sorted_vals = sorted(vals)
    idx = max(0, int(len(sorted_vals) * 0.95) - 1)
    return round(sorted_vals[idx], 2)


def _aggregate(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    stop_vals = [r["latency_after_stop_ms"] for r in rounds]
    start_vals = [r["latency_after_start_ms"] for r in rounds]
    ratios = [r["ratio"] for r in rounds if r.get("ratio") is not None]
    return {
        "avg_after_stop_ms": round(statistics.mean(stop_vals), 2) if stop_vals else 0,
        "avg_after_start_ms": round(statistics.mean(start_vals), 2) if start_vals else 0,
        "median_after_stop_ms": round(statistics.median(stop_vals), 2) if stop_vals else 0,
        "median_after_start_ms": round(statistics.median(start_vals), 2) if start_vals else 0,
        "p95_after_stop_ms": _p95(stop_vals),
        "p95_after_start_ms": _p95(start_vals),
        "ratio_avg": round(statistics.mean(ratios), 3) if ratios else None,
        "rounds_stop_slower_count": sum(1 for r in rounds if r.get("stop_slower")),
        "rounds_total": len(rounds),
    }


def _step_by_name(basic: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for step in basic:
        if step.get("name") == name:
            return step
    return None


def _evaluate(
    basic: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    agg: dict[str, Any],
    stop_slower_threshold: float,
) -> dict[str, Any]:
    issues: list[str] = []

    for step in basic:
        if not _ok_step(step):
            issues.append(
                f"basic step failed: {step['name']} -> {step.get('error') or step.get('status_code')}"
            )

    for r in rounds:
        for key in ("scenario_a", "scenario_b"):
            for sub in r[key].values():
                if not _ok_step(sub):
                    issues.append(
                        f"round {r['round']} {key} failed: {sub.get('error') or sub.get('status_code')}"
                    )

        stop_body = r["scenario_a"]["stop"].get("body") or {}
        if stop_body.get("loaded") is not False:
            issues.append(f"round {r['round']} scenario_a stop loaded != false")

        start_body = r["scenario_b"]["start"].get("body") or {}
        if start_body.get("loaded") is not True:
            issues.append(f"round {r['round']} scenario_b start loaded != true")

    stop_slower_ratio = (
        agg["rounds_stop_slower_count"] / agg["rounds_total"] if agg["rounds_total"] else 0.0
    )
    latency_pass = stop_slower_ratio >= stop_slower_threshold

    ratio_avg = agg.get("ratio_avg") or 0.0
    latency_gap_significant = ratio_avg >= 10.0
    if not latency_pass:
        issues.append(
            f"latency comparison weak: only {agg['rounds_stop_slower_count']}/{agg['rounds_total']} "
            f"rounds have stop->transcribe slower than start->transcribe "
            f"(threshold {stop_slower_threshold:.0%})"
        )
    if latency_pass and not latency_gap_significant:
        issues.append(
            f"latency gap not significant enough for manual inspection: ratio_avg={ratio_avg} (<10)"
        )

    functional_pass = not any("failed" in i or "loaded !=" in i for i in issues)
    if functional_pass and latency_pass and latency_gap_significant:
        judgement = "PASS"
        conclusion = (
            "start/stop functional checks passed; stop->transcribe is significantly slower than "
            f"start->transcribe (ratio_avg={ratio_avg})."
        )
    elif functional_pass and latency_pass:
        judgement = "WEAK_PASS"
        conclusion = (
            "Functional checks passed and stop is slower, but latency gap is too small to confirm "
            "cold-reload behavior (expected large gap like baseline report)."
        )
    elif functional_pass:
        judgement = "FAIL"
        conclusion = "Functional checks passed but latency difference is not significant enough."
    else:
        judgement = "FAIL"
        conclusion = "Functional or connectivity failures detected; see issues."

    return {
        "functional_pass": functional_pass,
        "latency_pass": latency_pass,
        "latency_gap_significant": latency_gap_significant,
        "stop_slower_ratio": round(stop_slower_ratio, 3),
        "stop_slower_threshold": stop_slower_threshold,
        "issues": issues,
        "judgement": judgement,
        "conclusion": conclusion,
    }


def _load_baseline_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_metrics(report: dict[str, Any]) -> dict[str, Any]:
    basic = report.get("basic_checks", [])
    agg = report.get("aggregate", {})
    basic_start = _step_by_name(basic, "basic_start")
    basic_implicit = _step_by_name(basic, "basic_transcribe_implicit")
    return {
        "timestamp": report.get("meta", {}).get("timestamp"),
        "basic_start_ms": basic_start.get("elapsed_ms") if basic_start else None,
        "basic_transcribe_implicit_ms": (
            basic_implicit.get("elapsed_ms") if basic_implicit else None
        ),
        "avg_after_stop_ms": agg.get("avg_after_stop_ms"),
        "avg_after_start_ms": agg.get("avg_after_start_ms"),
        "ratio_avg": agg.get("ratio_avg"),
        "judgement": report.get("evaluation", {}).get("judgement"),
    }


def _build_investigation(
    *,
    current_report: dict[str, Any],
    current_json_path: Path,
    baseline_report: dict[str, Any] | None,
    lifecycle: dict[str, Any],
    env_snapshot: dict[str, str | None],
    model_layout: dict[str, Any],
    git_snapshot: dict[str, Any],
) -> dict[str, Any]:
    current_metrics = _extract_metrics(current_report)
    baseline_metrics = _extract_metrics(baseline_report) if baseline_report else None

    hypotheses: list[dict[str, str]] = []
    changed_files = "\n".join(git_snapshot.get("all_changed_files", []))
    if "src/asr_server/service/whisper_model.py" in changed_files:
        hypotheses.append(
            {
                "area": "whisper_model.py",
                "detail": (
                    "Model init now resolves local/HF path via ensure_whisper_model_path and uses "
                    "local_files_only=True. If resolved path is local/HF cache, cold reload after "
                    "stop may become much faster than baseline."
                ),
            }
        )
    if "src/asr_server/utils/whisper_assets.py" in changed_files:
        hypotheses.append(
            {
                "area": "whisper_assets.py",
                "detail": (
                    "New model path resolution may prefer .models or HF cache, changing disk I/O "
                    "profile and reducing stop->transcribe reload penalty."
                ),
            }
        )
    if "src/asr_server/main.py" in changed_files:
        hypotheses.append(
            {
                "area": "main.py",
                "detail": "Startup now loads .env via _load_dotenv; environment differences may "
                "alter model/device selection across runs.",
            }
        )

    ratio_current = current_metrics.get("ratio_avg") or 0.0
    ratio_baseline = (baseline_metrics or {}).get("ratio_avg") or 0.0
    gap_regression = ratio_current < 10.0 and ratio_baseline >= 10.0

    conclusion = (
        "Current uncommitted changes show a much smaller stop/start transcribe gap than baseline "
        f"({ratio_current} vs {ratio_baseline}). This suggests start/stop API semantics still pass, "
        "but cold-reload cost after stop is no longer obvious."
    )
    if gap_regression and model_layout.get("resolved_source") in {"local", "hf_cache"}:
        conclusion += (
            f" Likely linked to faster local model resolution (source={model_layout.get('resolved_source')})."
        )

    return {
        "meta": {
            "timestamp": _now_stamp(),
            "baseline_report_json": str(BASELINE_REPORT_JSON),
            "current_report_json": str(current_json_path),
            "baseline_commit": BASELINE_COMMIT,
        },
        "server_lifecycle": lifecycle,
        "environment": env_snapshot,
        "model_layout": model_layout,
        "git_snapshot": git_snapshot,
        "metrics": {
            "baseline": baseline_metrics,
            "current": current_metrics,
        },
        "comparison": {
            "ratio_avg_delta": round(ratio_current - ratio_baseline, 3),
            "avg_after_stop_ms_delta": round(
                (current_metrics.get("avg_after_stop_ms") or 0)
                - (baseline_metrics or {}).get("avg_after_stop_ms", 0),
                2,
            ),
            "gap_regression": gap_regression,
        },
        "hypotheses": hypotheses,
        "conclusion": conclusion,
    }


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    agg = report["aggregate"]
    ev = report["evaluation"]
    lines = [
        "# start/stop Check Summary",
        "",
        f"- **Timestamp**: {report['meta']['timestamp']}",
        f"- **Base URL**: `{report['meta']['base_url']}`",
        f"- **WAV**: `{report['meta']['wav']}`",
        f"- **Rounds**: {report['meta']['rounds']}",
        f"- **Judgement**: **{ev['judgement']}**",
        f"- **Conclusion**: {ev['conclusion']}",
        "",
        "## Aggregate Latency",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| avg_after_stop_ms | {agg['avg_after_stop_ms']} |",
        f"| avg_after_start_ms | {agg['avg_after_start_ms']} |",
        f"| median_after_stop_ms | {agg['median_after_stop_ms']} |",
        f"| median_after_start_ms | {agg['median_after_start_ms']} |",
        f"| p95_after_stop_ms | {agg['p95_after_stop_ms']} |",
        f"| p95_after_start_ms | {agg['p95_after_start_ms']} |",
        f"| ratio_avg | {agg['ratio_avg']} |",
        f"| rounds_stop_slower | {agg['rounds_stop_slower_count']}/{agg['rounds_total']} |",
        "",
        "## Round Details",
        "",
        "| Round | after_stop_ms | after_start_ms | ratio | stop_slower |",
        "|-------|---------------|----------------|-------|-------------|",
    ]
    for r in report["rounds"]:
        lines.append(
            f"| {r['round']} | {r['latency_after_stop_ms']} | {r['latency_after_start_ms']} | "
            f"{r['ratio']} | {r['stop_slower']} |"
        )

    if ev["issues"]:
        lines.extend(["", "## Issues", ""])
        for issue in ev["issues"]:
            lines.append(f"- {issue}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_investigation_summary(path: Path, investigation: dict[str, Any]) -> None:
    baseline = investigation["metrics"]["baseline"] or {}
    current = investigation["metrics"]["current"]
    cmp_ = investigation["comparison"]
    lines = [
        "# start/stop Investigation Summary",
        "",
        f"- **Timestamp**: {investigation['meta']['timestamp']}",
        f"- **Baseline report**: `{investigation['meta']['baseline_report_json']}`",
        f"- **Current report**: `{investigation['meta']['current_report_json']}`",
        f"- **Baseline commit**: `{investigation['meta']['baseline_commit']}`",
        "",
        "## Key Metrics",
        "",
        "| Metric | Baseline | Current | Delta |",
        "|--------|----------|---------|-------|",
        f"| ratio_avg | {baseline.get('ratio_avg')} | {current.get('ratio_avg')} | {cmp_.get('ratio_avg_delta')} |",
        f"| avg_after_stop_ms | {baseline.get('avg_after_stop_ms')} | {current.get('avg_after_stop_ms')} | {cmp_.get('avg_after_stop_ms_delta')} |",
        f"| avg_after_start_ms | {baseline.get('avg_after_start_ms')} | {current.get('avg_after_start_ms')} | - |",
        f"| basic_start_ms | {baseline.get('basic_start_ms')} | {current.get('basic_start_ms')} | - |",
        f"| basic_transcribe_implicit_ms | {baseline.get('basic_transcribe_implicit_ms')} | {current.get('basic_transcribe_implicit_ms')} | - |",
        "",
        "## Server Lifecycle",
        "",
        f"- managed: {investigation['server_lifecycle'].get('managed')}",
        f"- port_guard: {json.dumps(investigation['server_lifecycle'].get('port_guard'), ensure_ascii=False)}",
        "",
        "## Model Layout",
        "",
        f"```json\n{json.dumps(investigation['model_layout'], ensure_ascii=False, indent=2)}\n```",
        "",
        "## Git Snapshot",
        "",
        f"- HEAD: `{investigation['git_snapshot'].get('head')}`",
        f"- changed files since baseline:",
    ]
    for name in investigation["git_snapshot"].get("all_changed_files", []):
        lines.append(f"  - `{name}`")

    lines.extend(["", "## Hypotheses", ""])
    for item in investigation["hypotheses"]:
        lines.append(f"- **{item['area']}**: {item['detail']}")

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            investigation["conclusion"],
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    wav = _resolve_wav(args)
    base_url = args.base_url.rstrip("/")
    transcribe_url = f"{base_url}/api/transcribe"
    start_url = f"{transcribe_url}/start"
    stop_url = f"{transcribe_url}/stop"

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now_stamp()
    json_path = args.reports_dir / f"start_stop_check_{stamp}.json"
    md_path = args.reports_dir / f"start_stop_check_{stamp}_summary.md"
    investigation_json_path = args.reports_dir / f"start_stop_investigation_{stamp}.json"
    investigation_md_path = args.reports_dir / f"start_stop_investigation_{stamp}_summary.md"

    env_snapshot = _collect_env_snapshot()
    model_layout = _collect_model_layout()
    git_snapshot = _git_snapshot()

    print(f"wav: {wav.name}")
    print(f"rounds: {args.rounds}")

    t0 = time.perf_counter()
    lifecycle: dict[str, Any] = {"managed": False}
    try:
        with _managed_server(base_url, manage_server=not args.no_manage_server) as lifecycle:
            with httpx.Client(timeout=args.timeout) as client:
                basic = _run_basic_checks(client, transcribe_url, start_url, stop_url, wav)
                rounds = [
                    _run_latency_round(client, transcribe_url, start_url, stop_url, wav, i + 1)
                    for i in range(args.rounds)
                ]
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    aggregate = _aggregate(rounds)
    evaluation = _evaluate(basic, rounds, aggregate, args.stop_slower_ratio)

    report = {
        "meta": {
            "timestamp": stamp,
            "base_url": base_url,
            "wav": wav.name,
            "wav_path": str(wav),
            "rounds": args.rounds,
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "environment": env_snapshot,
            "model_layout": model_layout,
            "server_lifecycle": lifecycle,
            "git_snapshot": git_snapshot,
        },
        "basic_checks": basic,
        "rounds": rounds,
        "aggregate": aggregate,
        "evaluation": evaluation,
    }

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(md_path, report)

    baseline_report = _load_baseline_report(BASELINE_REPORT_JSON)
    investigation = _build_investigation(
        current_report=report,
        current_json_path=json_path,
        baseline_report=baseline_report,
        lifecycle=lifecycle,
        env_snapshot=env_snapshot,
        model_layout=model_layout,
        git_snapshot=git_snapshot,
    )
    investigation_json_path.write_text(
        json.dumps(investigation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_investigation_summary(investigation_md_path, investigation)

    print(f"judgement: {evaluation['judgement']}")
    print(f"json: {json_path}")
    print(f"summary: {md_path}")
    print(f"investigation_json: {investigation_json_path}")
    print(f"investigation_summary: {investigation_md_path}")
    return 0 if evaluation["judgement"] in {"PASS", "WEAK_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
