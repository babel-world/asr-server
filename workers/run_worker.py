"""
Dispatch worker CLI commands via registry.toml and uv run.
Optimized for FastAPI subprocess invocation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TypedDict

# 确保以脚本所在目录为基准，不受 FastAPI 执行时的 cwd 影响
WORKERS_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = WORKERS_DIR / "registry.toml"

class WorkerEntry(TypedDict):
    path: str
    cli: str


def load_registry() -> dict[str, WorkerEntry]:
    """加载并校验 registry.toml 的存在性"""
    if not REGISTRY_PATH.is_file():
        raise SystemExit(f"❌ 致命错误: 找不到注册表配置文件: {REGISTRY_PATH}")

    try:
        with REGISTRY_PATH.open("rb") as file:
            return tomllib.load(file)
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"❌ 致命错误: registry.toml 格式损坏。原因: {e}")
    except OSError as e:
        raise SystemExit(f"❌ 致命错误: 读取 registry.toml 失败。原因: {e}")

def resolve_worker(alias: str) -> tuple[Path, str]:
    """解析别名并严格校验 worker 目录和配置的完整性"""
    registry = load_registry()
    
    if alias not in registry:
        known = "\n  - ".join([""] + sorted(registry.keys()))
        raise SystemExit(
            f"❌ 错误: 未知的 worker 别名 '{alias}'。\n"
            f"已注册的 workers:{known}"
        )

    entry = registry[alias]
    
    # 校验 TOML 中是否包含必要的字段
    if "path" not in entry or "cli" not in entry:
        raise SystemExit(
            f"❌ 错误: registry.toml 中 [{alias}] 缺失必要的 'path' 或 'cli' 字段。"
        )

    if not isinstance(entry["path"], str) or not isinstance(entry["cli"], str):
        raise SystemExit(
            f"❌ 错误: registry.toml 中 [{alias}] 的 'path' 和 'cli' 必须是字符串。"
        )

    worker_dir = (WORKERS_DIR / entry["path"]).resolve()
    if not worker_dir.is_relative_to(WORKERS_DIR):
        raise SystemExit(
            f"❌ 错误: registry.toml 中 [{alias}] 的 path 越界: {entry['path']}"
        )
    
    # 校验工作目录与项目隔离文件
    if not worker_dir.is_dir():
        raise SystemExit(f"❌ 错误: Worker 目录不存在: {worker_dir}")
        
    if not (worker_dir / "pyproject.toml").is_file():
        raise SystemExit(f"❌ 错误: 目标目录缺乏 pyproject.toml，不是一个合法的 uv 项目: {worker_dir}")

    return worker_dir, entry["cli"]

def run_worker(alias: str, worker_args: list[str]) -> int:
    """使用 uv run 启动真正的 worker 进程"""
    if shutil.which("uv") is None:
        raise SystemExit("❌ 致命错误: 系统环境变量 PATH 中未找到 'uv'。请先安装 uv。")

    worker_dir, cli_entry = resolve_worker(alias)
    
    # 构建 uv 执行命令
    command = ["uv", "run", cli_entry, *worker_args]
    
    # ---------------------------------------------------------
    # 核心优化：进程替换 (Process Replacement)
    # ---------------------------------------------------------
    # 如果是在 Linux/macOS 服务器上运行 (FastAPI 生产环境常见部署)
    # 使用 os.execvp 完美解决进程套娃问题。
    # 此时 `python run_worker.py` 这个进程会直接被 `uv run` 替换掉，
    # FastAPI 直接监控 `uv run`，占用资源更少，停止任务时也没有僵尸进程。
    if os.name == "posix":
        try:
            os.chdir(worker_dir)  # 切换到目标环境目录
            # execvp 的第一个参数是可执行文件，第二个参数是完整的 argv 列表
            os.execvp("uv", command)
        except KeyboardInterrupt:
            return 130
        except OSError as e:
            print(f"❌ 执行异常: {e}", file=sys.stderr)
            return 1
        
    # 如果是在 Windows 上开发 (os.execvp 行为不完美，回退到标准 subprocess)
    else:
        try:
            # 这里的 cwd=worker_dir 实现了极其干净的隔离
            result = subprocess.run(command, cwd=worker_dir, check=False)
            return result.returncode
        except KeyboardInterrupt:
            # 妥善处理 FastAPI 发送的终止信号 (Ctrl+C)
            return 130
        except OSError as e:
            print(f"❌ 执行异常: {e}", file=sys.stderr)
            return 1

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a registered model worker CLI via uv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example registry.toml format:
  [chinese-hubert-base]
  path = "huggingface/TencentGameMate/chinese-hubert-base"
  cli = "hubert-feature-extract"
"""
    )
    parser.add_argument("alias", help="Worker alias defined in registry.toml")
    parser.add_argument(
        "worker_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the worker CLI. Use '--' to separate args.",
    )
    
    args = parser.parse_args(argv)

    # 规范化参数传递：过滤掉 FastAPI 调用时可能传入的 '--' 占位符
    worker_args = args.worker_args
    if worker_args and worker_args[0] == "--":
        worker_args = worker_args[1:]

    return run_worker(args.alias, worker_args)

if __name__ == "__main__":
    sys.exit(main())