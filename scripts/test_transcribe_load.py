"""对 POST /api/transcribe 做并发压测。

用法（项目根目录）::

    uv sync --extra dev
    uv run python scripts/test_transcribe_load.py
    uv run python scripts/test_transcribe_load.py --smoke
    uv run python scripts/test_transcribe_load.py --requests-per-level 30

报告输出目录：TEMP/reports/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np

DEFAULT_BASE_URL = "http://127.0.0.1:19031/api/transcribe"
CONCURRENCY_LEVELS = [1, 2, 5, 10, 20, 50]
DEFAULT_REQUESTS_PER_LEVEL = 60
DEFAULT_TIMEOUT_S = 600.0


@dataclass
class RequestRecord:
    index: int
    wav_path: str
    success: bool
    status_code: int | None
    latency_ms: float
    error_type: str | None = None
    error_detail: str | None = None
    language: str | None = None
    text_len: int | None = None


@dataclass
class LevelSummary:
    concurrency: int
    total: int
    success: int
    fail: int
    success_rate: float
    wall_time_ms: float
    requests_per_second: float
    latency_ms: dict[str, float]
    errors_by_type: dict[str, int] = field(default_factory=dict)


def _discover_wav_files(sample_dir: Path) -> list[Path]:
    files = sorted(sample_dir.glob("**/*.wav"))
    if not files:
        raise FileNotFoundError(f"未在 {sample_dir} 中找到 .wav 文件")
    return files


def _percentiles_ms(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.array(latencies, dtype=np.float64)
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


async def _post_one(
    client: httpx.AsyncClient,
    url: str,
    wav_path: Path,
    timeout_s: float,
) -> tuple[bool, int | None, float, str | None, str | None, str | None, int | None]:
    started = time.perf_counter()
    try:
        with wav_path.open("rb") as f:
            response = await client.post(
                url,
                files={"file": (wav_path.name, f, "audio/wav")},
                timeout=timeout_s,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code == 200:
            body = response.json()
            text = body.get("transcribedText") or body.get("transcribed_text") or ""
            lang = body.get("language")
            return True, response.status_code, latency_ms, None, None, lang, len(text)
        detail = response.text[:500] if response.text else response.reason_phrase
        return (
            False,
            response.status_code,
            latency_ms,
            "http_error",
            detail,
            None,
            None,
        )
    except httpx.TimeoutException as e:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return False, None, latency_ms, "timeout", str(e), None, None
    except httpx.ConnectError as e:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return False, None, latency_ms, "connect_error", str(e), None, None
    except Exception as e:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return False, None, latency_ms, "other", repr(e), None, None


async def _run_level(
    *,
    url: str,
    wav_files: list[Path],
    concurrency: int,
    num_requests: int,
    timeout_s: float,
    rng: random.Random,
) -> tuple[list[RequestRecord], LevelSummary]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded_request(
        client: httpx.AsyncClient, index: int
    ) -> RequestRecord:
        wav_path = wav_files[index % len(wav_files)]
        if rng.random() < 0.5:
            wav_path = rng.choice(wav_files)
        async with semaphore:
            (
                ok,
                status,
                latency_ms,
                err_type,
                err_detail,
                lang,
                text_len,
            ) = await _post_one(client, url, wav_path, timeout_s)
        return RequestRecord(
            index=index,
            wav_path=str(wav_path),
            success=ok,
            status_code=status,
            latency_ms=latency_ms,
            error_type=err_type,
            error_detail=err_detail,
            language=lang,
            text_len=text_len,
        )

    records: list[RequestRecord] = []
    wall_start = time.perf_counter()
    limits = httpx.Limits(max_connections=concurrency + 5, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [_bounded_request(client, i) for i in range(num_requests)]
        records = list(await asyncio.gather(*tasks))
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    success_records = [r for r in records if r.success]
    fail_records = [r for r in records if not r.success]
    latencies = [r.latency_ms for r in records]
    errors_by_type: dict[str, int] = {}
    for r in fail_records:
        key = r.error_type or "unknown"
        errors_by_type[key] = errors_by_type.get(key, 0) + 1

    summary = LevelSummary(
        concurrency=concurrency,
        total=len(records),
        success=len(success_records),
        fail=len(fail_records),
        success_rate=len(success_records) / len(records) if records else 0.0,
        wall_time_ms=wall_ms,
        requests_per_second=(len(records) / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0,
        latency_ms=_percentiles_ms(latencies),
        errors_by_type=errors_by_type,
    )
    return records, summary


async def _smoke_test(url: str, wav_files: list[Path], timeout_s: float) -> bool:
    print(f"冒烟测试: {url}，样本数 {min(3, len(wav_files))}")
    async with httpx.AsyncClient() as client:
        for wav in wav_files[:3]:
            ok, status, latency_ms, err_type, err_detail, lang, text_len = await _post_one(
                client, url, wav, timeout_s
            )
            status_s = status if status is not None else "-"
            if ok:
                print(f"  OK {wav.name} status={status_s} {latency_ms:.0f}ms lang={lang} text_len={text_len}")
            else:
                print(
                    f"  FAIL {wav.name} status={status_s} {latency_ms:.0f}ms "
                    f"type={err_type} detail={err_detail}",
                    file=sys.stderr,
                )
                return False
    return True


def _print_summary_table(summaries: list[LevelSummary]) -> None:
    print()
    print("并发压测汇总")
    print("-" * 88)
    print(
        f"{'并发':>4} {'成功':>6} {'失败':>6} {'成功率':>8} "
        f"{'RPS':>8} {'avg_ms':>10} {'p95_ms':>10} {'max_ms':>10}"
    )
    print("-" * 88)
    for s in summaries:
        lat = s.latency_ms
        print(
            f"{s.concurrency:>4} {s.success:>6} {s.fail:>6} {s.success_rate:>7.1%} "
            f"{s.requests_per_second:>8.2f} {lat['avg']:>10.0f} {lat['p95']:>10.0f} {lat['max']:>10.0f}"
        )
    print("-" * 88)


def _build_report(
    *,
    url: str,
    sample_dir: Path,
    num_samples: int,
    requests_per_level: int,
    summaries: list[LevelSummary],
    all_records: dict[int, list[RequestRecord]],
) -> dict[str, Any]:
    return {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "url": url,
            "sample_dir": str(sample_dir),
            "wav_count": num_samples,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "requests_per_level": requests_per_level,
            "note": "服务端 transcribe 使用全局 asyncio.Lock，转录实际串行执行",
        },
        "summaries": [asdict(s) for s in summaries],
        "records_by_concurrency": {
            str(k): [asdict(r) for r in v] for k, v in all_records.items()
        },
    }


async def _async_main(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    sample_dir = Path(args.sample_dir)
    if not sample_dir.is_absolute():
        sample_dir = project_root / sample_dir

    try:
        wav_files = _discover_wav_files(sample_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    url = args.url.rstrip("/")
    if not url.endswith("/transcribe"):
        if url.endswith("/api"):
            url = f"{url}/transcribe"
        elif "/api/transcribe" not in url:
            url = f"{url}/api/transcribe"

    print(f"样本目录: {sample_dir} ({len(wav_files)} 个 wav)")
    print(f"目标 URL: {url}")

    if not await _smoke_test(url, wav_files, args.timeout):
        print("冒烟测试失败，请确认 asr-server 已启动且可访问。", file=sys.stderr)
        return 1

    if args.smoke:
        print("仅冒烟模式，已退出。")
        return 0

    rng = random.Random(args.seed)
    levels = CONCURRENCY_LEVELS if not args.levels else args.levels
    summaries: list[LevelSummary] = []
    all_records: dict[int, list[RequestRecord]] = {}

    for concurrency in levels:
        print()
        print(f"=== 并发 {concurrency}，请求数 {args.requests_per_level} ===")
        records, summary = await _run_level(
            url=url,
            wav_files=wav_files,
            concurrency=concurrency,
            num_requests=args.requests_per_level,
            timeout_s=args.timeout,
            rng=rng,
        )
        all_records[concurrency] = records
        summaries.append(summary)
        lat = summary.latency_ms
        print(
            f"完成: 成功 {summary.success}/{summary.total} "
            f"RPS={summary.requests_per_second:.2f} "
            f"avg={lat['avg']:.0f}ms p95={lat['p95']:.0f}ms max={lat['max']:.0f}ms"
        )
        if summary.errors_by_type:
            print(f"错误分布: {summary.errors_by_type}")

    _print_summary_table(summaries)

    report = _build_report(
        url=url,
        sample_dir=sample_dir,
        num_samples=len(wav_files),
        requests_per_level=args.requests_per_level,
        summaries=summaries,
        all_records=all_records,
    )

    reports_dir = sample_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = reports_dir / f"transcribe_load_{stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"报告已写入: {out_path.relative_to(project_root)}")

    _print_analysis(summaries)
    return 0


def _print_analysis(summaries: list[LevelSummary]) -> None:
    if len(summaries) < 2:
        return
    print()
    print("简要分析")
    print("-" * 60)
    baseline = summaries[0]
    base_rps = baseline.requests_per_second
    base_p95 = baseline.latency_ms["p95"]
    print(
        f"- 基线（并发 {baseline.concurrency}）: RPS≈{base_rps:.2f}, p95≈{base_p95:.0f}ms"
    )
    print("- 服务端全局锁使转录串行，并发升高时 RPS 通常不会线性增长，时延会排队拉长。")
    for s in summaries[1:]:
        rps_ratio = s.requests_per_second / base_rps if base_rps > 0 else 0.0
        p95_ratio = s.latency_ms["p95"] / base_p95 if base_p95 > 0 else 0.0
        print(
            f"- 并发 {s.concurrency}: RPS 相对基线 {rps_ratio:.2f}x, "
            f"p95 相对基线 {p95_ratio:.2f}x, 成功率 {s.success_rate:.1%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="POST /api/transcribe 并发压测")
    parser.add_argument(
        "--url",
        default=DEFAULT_BASE_URL,
        help=f"转录接口 URL（默认 {DEFAULT_BASE_URL}）",
    )
    parser.add_argument(
        "--sample-dir",
        default="TEMP",
        help="wav 样本目录（默认 TEMP）",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=DEFAULT_REQUESTS_PER_LEVEL,
        help=f"每个并发档位发送的请求数（默认 {DEFAULT_REQUESTS_PER_LEVEL}）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"单请求超时秒数（默认 {DEFAULT_TIMEOUT_S}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机抽样的种子",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="仅执行冒烟测试（3 个请求）后退出",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="自定义并发档位，例如 --levels 1 5 10",
    )
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
