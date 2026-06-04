"""验证 POST /api/transcribe/start 与 /api/transcribe/stop 的行为与时延差异。

测试前请自行启动服务（项目根目录）::

    uv run asr-server

然后执行本脚本::

    uv sync --extra dev
    uv run python scripts/test_start_stop.py
    uv run python scripts/test_start_stop.py --rounds 3
    uv run python scripts/test_start_stop.py --wav TEMP/manbo_0035_0006602240-0006684160.wav

报告输出目录：TEMP/reports/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "http://127.0.0.1:19031"
DEFAULT_SAMPLE_DIR = REPO_ROOT / "TEMP"
DEFAULT_REPORTS_DIR = REPO_ROOT / "TEMP" / "reports"
DEFAULT_WAV = DEFAULT_SAMPLE_DIR / "manbo_0035_0006602240-0006684160.wav"
DEFAULT_ROUNDS = 5
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_STOP_SLOWER_RATIO = 0.8


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
        default=DEFAULT_WAV,
        help="WAV file for transcribe requests",
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
    return parser.parse_args()


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_wav(args: argparse.Namespace) -> Path:
    wav = args.wav
    if wav.exists():
        return wav.resolve()
    if args.sample_dir.exists():
        candidates = sorted(args.sample_dir.glob("*.wav"))
        if candidates:
            return candidates[0].resolve()
    print(f"WAV not found: {wav}", file=sys.stderr)
    raise SystemExit(1)


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
    if not latency_pass:
        issues.append(
            f"latency comparison weak: only {agg['rounds_stop_slower_count']}/{agg['rounds_total']} "
            f"rounds have stop->transcribe slower than start->transcribe "
            f"(threshold {stop_slower_threshold:.0%})"
        )

    functional_pass = not any("failed" in i or "loaded !=" in i for i in issues)
    if functional_pass and latency_pass:
        judgement = "PASS"
        conclusion = (
            "start/stop functional checks passed; start preloads model and stop triggers reload cost "
            f"({agg['rounds_stop_slower_count']}/{agg['rounds_total']} rounds slower after stop)."
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
        "stop_slower_ratio": round(stop_slower_ratio, 3),
        "stop_slower_threshold": stop_slower_threshold,
        "issues": issues,
        "judgement": judgement,
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

    print(f"wav: {wav.name}")
    print(f"rounds: {args.rounds}")

    t0 = time.perf_counter()
    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{base_url}/docs")
        if health.status_code != 200:
            print(
                f"Server not ready at {base_url}. Start it first: uv run asr-server",
                file=sys.stderr,
            )
            print(f"/docs -> {health.status_code}", file=sys.stderr)
            return 1

        basic = _run_basic_checks(client, transcribe_url, start_url, stop_url, wav)
        rounds = [
            _run_latency_round(client, transcribe_url, start_url, stop_url, wav, i + 1)
            for i in range(args.rounds)
        ]

    aggregate = _aggregate(rounds)
    evaluation = _evaluate(basic, rounds, aggregate, args.stop_slower_ratio)

    report = {
        "meta": {
            "timestamp": stamp,
            "base_url": base_url,
            "wav": wav.name,
            "rounds": args.rounds,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        },
        "basic_checks": basic,
        "rounds": rounds,
        "aggregate": aggregate,
        "evaluation": evaluation,
    }

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(md_path, report)

    print(f"judgement: {evaluation['judgement']}")
    print(f"json: {json_path}")
    print(f"summary: {md_path}")
    return 0 if evaluation["judgement"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
