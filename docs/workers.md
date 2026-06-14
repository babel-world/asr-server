# Worker 机制详解

本文档描述 asr-server 的 worker 子进程系统：设计动机、目录结构、注册表、调度层、IPC 协议、客户端 Session、错误体系、超时配置，以及新增 worker 的完整步骤。

---

## 目录

1. [设计动机](#1-设计动机)
2. [整体架构](#2-整体架构)
3. [目录结构](#3-目录结构)
4. [注册表 registry.toml](#4-注册表-registrytoml)
5. [调度器 run_worker.py](#5-调度器-run_workerpy)
6. [Worker 子包结构](#6-worker-子包结构)
   - 6.1 [download.py — 模型解析与下载](#61-downloadpy--模型解析与下载)
   - 6.2 [extract.py — 模型加载与推理](#62-extractpy--模型加载与推理)
   - 6.3 [serve.py — 长驻服务循环](#63-servepy--长驻服务循环)
   - 6.4 [cli.py — CLI 入口](#64-clipy--cli-入口)
7. [stdin/stdout JSON Lines 协议](#7-stdinstdout-json-lines-协议)
8. [客户端基础设施 infra/worker](#8-客户端基础设施-infraworker)
   - 8.1 [config.py — 超时解析](#81-configpy--超时解析)
   - 8.2 [errors.py — 错误体系](#82-errorspy--错误体系)
   - 8.3 [dispatch.py — 一次性派发](#83-dispatchpy--一次性派发)
   - 8.4 [session.py — 长驻 Session](#84-sessionpy--长驻-session)
9. [Service 层与 HTTP 层](#9-service-层与-http-层)
10. [完整调用链（端到端）](#10-完整调用链端到端)
11. [错误 → HTTP 状态码映射](#11-错误--http-状态码映射)
12. [并发与线程安全](#12-并发与线程安全)
13. [平台差异（Linux vs Windows）](#13-平台差异linux-vs-windows)
14. [已注册的 Workers](#14-已注册的-workers)
    - 14.1 [chinese-hubert-base](#141-chinese-hubert-base)
    - 14.2 [speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common](#142-speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common)
15. [本地开发与调试](#15-本地开发与调试)
16. [如何新增一个 Worker](#16-如何新增一个-worker)

---

## 1. 设计动机

ML 推理进程（torch、transformers、modelscope 等）有三个特点与 FastAPI 主进程天然冲突：

| 问题 | 影响 |
|------|------|
| 依赖重型、版本严格，与主服务难以共存于同一 Python 环境 | 依赖冲突，安装复杂 |
| 模型加载耗时数十秒、占用数 GB 显存 | 每次请求都冷启动不可接受 |
| CUDA 上下文一旦建立，进程间难以共享 | 多进程/线程推理要特殊处理 |

解决方案：**将每个 ML 模型拆分为独立的 uv 子项目（worker），通过 subprocess 隔离运行，使用 stdin/stdout JSON Lines 与主服务通信。**

---

## 2. 整体架构

```
HTTP 请求
    │
    ▼
FastAPI 路由层  (api/routes/features.py)
    │
    ▼
Service 层     (service/features/<worker>.py)
  asyncio.Lock  ← 串行化并发请求
  asyncio.to_thread  ← 不阻塞事件循环
    │
    ▼
infra/worker/session.py
  PersistentWorkerSession
    │  _lock (threading.Lock) ← 保护 stdin/stdout 读写
    │
    ├─── 首次/崩溃后 ───► subprocess.Popen
    │                        uv run <cli> serve
    │                        cwd = workers/<path>/
    │
    │  stdin ──JSON Line──►  serve.py (长驻循环)
    │  stdout ◄─JSON Line──    ↕ extract.py
    │  stderr ──► _StderrDrainer (后台线程)
    │
    ▼
临时目录 input.npy / output.npy
    │
    ▼
FileResponse 返回给 HTTP 客户端
```

---

## 3. 目录结构

```
asr-server/
├── workers/
│   ├── registry.toml                        # 所有 worker 的注册表
│   ├── run_worker.py                        # 调度器（读 registry → uv run）
│   │
│   ├── huggingface/
│   │   └── TencentGameMate/
│   │       └── chinese-hubert-base/         # Worker A：HuBERT 特征提取
│   │           ├── pyproject.toml
│   │           ├── uv.lock
│   │           ├── .python-version
│   │           └── src/chinese_hubert_base/
│   │               ├── __init__.py
│   │               ├── cli.py
│   │               ├── download.py
│   │               ├── extract.py
│   │               └── serve.py
│   │
│   └── modelscope/
│       └── iic/
│           └── speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/   # Worker B：SV Embedding
│               ├── pyproject.toml
│               ├── uv.lock
│               ├── .python-version
│               ├── smoke_test.py
│               └── src/speech_eres2netv2w24s4ep4_sv_zh_cn_16k_common/
│                   ├── __init__.py
│                   ├── cli.py
│                   ├── download.py
│                   ├── extract.py
│                   ├── serve.py
│                   └── vendor/eres2net/     # 内嵌 ERes2NetV2 实现
│
└── src/asr_server/
    └── infra/worker/                        # 客户端基础设施
        ├── __init__.py
        ├── config.py                        # 超时配置解析
        ├── errors.py                        # 异常类型
        ├── dispatch.py                      # 一次性派发（spawn_worker）
        └── session.py                       # 长驻 Session（PersistentWorkerSession）
```

---

## 4. 注册表 registry.toml

位置：`workers/registry.toml`

每个 TOML 表（`[alias]`）对应一个 worker，是整个系统的**唯一注册中心**。

```toml
[chinese-hubert-base]
source = "huggingface"
repo_id = "TencentGameMate/chinese-hubert-base"
path = "huggingface/TencentGameMate/chinese-hubert-base"
cli = "chinese-hubert-base"
description = "Chinese HuBERT feature extraction (scaffold)"
timeout_sec = 120

[speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common]
source = "modelscope"
repo_id = "iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common"
path = "modelscope/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common"
cli = "speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common"
description = "ERes2NetV2 SV embedding for GPT-SoVITS v2Pro pipeline B"
timeout_sec = 120
```

### 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `source` | 否 | 模型来源，仅作文档用途（`"huggingface"` / `"modelscope"`） |
| `repo_id` | 否 | 原始仓库 ID，仅作文档用途 |
| `path` | **是** | 相对 `workers/` 的目录路径，必须包含 `pyproject.toml` |
| `cli` | **是** | `pyproject.toml` 中 `[project.scripts]` 的脚本名，即 `uv run` 的目标 |
| `description` | 否 | 描述，仅文档用途 |
| `timeout_sec` | 否 | 该 worker 的超时秒数，覆盖全局默认值（见[超时配置](#81-configpy--超时配置)） |

### resolve_worker 的安全校验

`run_worker.py` 的 `resolve_worker(alias)` 在解析别名时做三道校验，防止路径逃逸或目录缺失：

1. alias 必须存在于 registry
2. `path` 解析后必须仍位于 `workers/` 目录之内（防止 `../../` 等路径攻击）
3. 目标目录必须存在，且必须包含 `pyproject.toml`

---

## 5. 调度器 run_worker.py

位置：`workers/run_worker.py`

这个脚本是服务端调度层的**核心桥接点**。它被客户端通过 `importlib` 动态加载，不需要安装到主服务的 Python 环境中。

### 核心函数

#### `build_worker_command(alias, worker_args) → (argv, cwd)`

将 alias + 参数 组装为 uv 命令：

```python
command = ["uv", "run", cli_entry, *worker_args]
cwd = workers/<path>/
```

例如，alias=`"chinese-hubert-base"`、args=`["serve"]` 时，产生：
```
cwd  = workers/huggingface/TencentGameMate/chinese-hubert-base/
argv = ["uv", "run", "chinese-hubert-base", "serve"]
```

uv 会自动在 `cwd` 下的 `.venv` 中解析并执行 CLI，确保每个 worker 使用自己独立的依赖环境。

#### `spawn_worker(alias, worker_args, *, timeout_sec) → CompletedProcess`

用 `subprocess.run` **同步等待**子进程结束。适合一次性任务（`extract`、`download`）。

```python
subprocess.run(command, cwd=worker_dir, capture_output=True, text=True, timeout=timeout_sec)
```

#### `run_worker(alias, worker_args) → int`

CLI 入口函数。在 POSIX 系统上使用 `os.execvp` 做**进程替换**（见[平台差异](#13-平台差异linux-vs-windows)）；在 Windows 上回退为 `subprocess.run`。

### 直接从命令行调用

```bash
# 在 asr-server 项目根目录下
uv run python workers/run_worker.py chinese-hubert-base -- extract --input foo.npy --output bar.npy
uv run python workers/run_worker.py speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common -- serve
```

`--` 之后的参数原样传递给 worker CLI。

---

## 6. Worker 子包结构

每个 worker 是一个标准的 uv 项目，包含四个核心模块。以 `chinese-hubert-base` 为例：

### 6.1 download.py — 模型解析与下载

职责：以**线程安全**的方式，按优先级解析或下载模型文件。

**解析优先级（chinese-hubert-base）：**

```
① workers/<path>/.models/chinese-hubert-base/   (worker 本地缓存，优先)
      ↓ 不存在或不完整
② HuggingFace Hub 本地缓存（local_files_only=True）
      ↓ 不存在
③ 从 HuggingFace Hub 实时下载 → 写入 ①
```

**解析优先级（speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common）：**

```
① workers/<path>/.models/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/   (优先)
      ↓ 不存在
② 从 ModelScope 实时下载 → 写入 ①
```

核心函数 `ensure_model_path()` 持有一个模块级 `threading.Lock`，保证并发调用时不会触发多次下载。

模型完整性校验：
- `chinese-hubert-base`：检查 `config.json` + `pytorch_model.bin`（> 1 MB）
- `speech-eres2netv2w24s4ep4`：检查 `pretrained_eres2netv2w24s4ep4.ckpt`（> 1 MB）

环境变量 `HF_TOKEN` 可用于访问私有 HuggingFace 仓库。

### 6.2 extract.py — 模型加载与推理

职责：懒加载模型到内存（只加载一次），并执行推理。

**核心设计：**

```python
_runtime_lock = threading.Lock()
_runtime: HubertRuntime | None = None  # 进程内单例

def get_runtime() -> HubertRuntime:
    """Return a cached runtime; reload when resolved model path changes."""
    model_path, _ = download.ensure_model_path()
    device, use_fp16 = resolve_device()
    with _runtime_lock:
        if _runtime is not None and _runtime.model_path == model_path ...:
            return _runtime  # 缓存命中
        _runtime = ...  # 重新加载
```

- 设备策略：检测到 CUDA 时使用 GPU + FP16；否则 CPU + FP32
- `chinese-hubert-base` 使用 `transformers.HubertModel`，直接接受 `input_values`（跳过 `Wav2Vec2FeatureExtractor`，遵循 GPT-SoVITS 约定）
- `speech-eres2netv2w24s4ep4` 使用内嵌的 `vendor/eres2net/ERes2NetV2.py`，推理前先提取 Kaldi Fbank 特征（80 mel bins）

**关键输入约定：**

| Worker | 输入格式 | 注意事项 |
|--------|----------|----------|
| `chinese-hubert-base` | float32, 16 kHz, mono, shape `(T,)` 或 `(1, T)` | GPT-SoVITS 特有的 1145.14 振幅标度 |
| `speech-eres2netv2w24s4ep4` | float32, 16 kHz, mono, shape `(T,)` 或 `(1, T)` | **常规振幅 [-1, 1]**，不得使用 1145.14 标度 |

### 6.3 serve.py — 长驻服务循环

职责：实现 `serve` 子命令，即 worker 进程的长驻模式。

```python
def run_serve() -> int:
    # 1. 启动时加载模型
    extract.get_runtime()

    # 2. 向 stdout 发送 ready 信号（客户端等待这一行）
    _write_response({"ok": True, "event": "ready"})

    # 3. 主循环：逐行读 stdin，写 stdout
    for raw_line in sys.stdin:
        payload = json.loads(raw_line.strip())
        if payload.get("cmd") == "shutdown":
            _write_response({"ok": True, "event": "shutdown"})
            return 0
        response = _handle_request(payload)
        _write_response(response)
```

模型只在进程启动时加载一次；`for raw_line in sys.stdin` 会阻塞等待，直到 stdin 被关闭（父进程终止）或收到 `shutdown` 命令。

### 6.4 cli.py — CLI 入口

提供三个子命令，统一由 `pyproject.toml` 中的 `[project.scripts]` 暴露：

| 子命令 | 用途 |
|--------|------|
| `download` | 解析或下载模型，打印 `model_path=` 和 `source=` |
| `extract --input <npy> --output <npy>` | 单次推理（开发/测试用，每次都冷启动） |
| `serve` | 长驻服务循环，供 asr-server 批量调用 |

```bash
# 在 worker 目录内，或通过 uv run 指定 cwd
uv run chinese-hubert-base download
uv run chinese-hubert-base extract --input wave.npy --output feat.npy
uv run chinese-hubert-base serve
```

---

## 7. stdin/stdout JSON Lines 协议

Worker 在 `serve` 模式下通过 **JSON Lines**（每行一个完整 JSON 对象）与父进程通信。所有消息均为 UTF-8 编码，以 `\n` 结尾，立即 `flush()`。

### 7.1 启动握手

```
父进程                              Worker 子进程
   │                                    │
   │── Popen(["uv","run",cli,"serve"]) ─►│
   │                                    │ 加载模型（数秒到数十秒）
   │◄─── {"ok": true, "event": "ready"} ─│  ← 成功
   │  或
   │◄─── {"ok": false, "error": "...", "event": "ready"} ─│  ← 失败，子进程退出
```

父进程最多等待 `SESSION_START_TIMEOUT_SEC = 300` 秒（5 分钟）。

### 7.2 推理请求

```
父进程                              Worker 子进程
   │── {"cmd": "extract",              │
   │    "input": "/tmp/xxx/input.npy", │
   │    "output": "/tmp/xxx/output.npy"}►│
   │                                    │ 推理，写文件
   │◄─── {"ok": true}                ───│  ← 成功
   │  或
   │◄─── {"ok": false, "error": "..."} ─│  ← 失败
```

input/output 均为**绝对路径**，通过文件系统交换数据（而非在管道中传输 blob）。

### 7.3 其他命令

```jsonc
// Ping（确认模型已加载）
→ {"cmd": "ping"}
← {"ok": true, "loaded": true}

// 优雅关闭
→ {"cmd": "shutdown"}
← {"ok": true, "event": "shutdown"}
// 之后子进程正常退出 (returncode=0)

// 未知命令
→ {"cmd": "foo"}
← {"ok": false, "error": "unknown cmd: 'foo'"}
```

### 7.4 关于 stderr

Worker 进程的 stderr **不参与协议**，仅用于调试日志（torch 输出、modelscope 进度条等）。客户端通过 `_StderrDrainer` 后台线程持续消费 stderr，防止管道缓冲区堆满导致死锁。错误发生时，会取最后 8192 字节（UTF-8）作为 `stderr_tail` 暴露给上层。

---

## 8. 客户端基础设施 infra/worker

### 8.1 config.py — 超时配置

`get_worker_timeout_sec(alias)` 按以下**优先级**解析超时秒数：

```
registry.toml 的 timeout_sec
    ↓ 未配置
环境变量 WORKER_SPAWN_TIMEOUT_<ALIAS_UPPER>
    例：alias="chinese-hubert-base" → WORKER_SPAWN_TIMEOUT_CHINESE_HUBERT_BASE
    ↓ 未设置
环境变量 WORKER_SPAWN_TIMEOUT_SEC
    ↓ 未设置
硬编码默认值 120.0 秒
```

alias 中的 `-` 被替换为 `_` 并转大写后拼接为环境变量名。

`_load_registry()` 使用 `@lru_cache(maxsize=1)` 缓存注册表，`clear_worker_config_cache()` 可在测试中清除缓存。

### 8.2 errors.py — 错误体系

```
WorkerSpawnError (base)
├── WorkerSpawnTimeout(alias, timeout_sec)
│     消息: "Worker '{alias}' timed out after {timeout_sec:g}s"
└── WorkerSpawnFailed(alias, returncode, stderr_tail)
      消息: "Worker '{alias}' failed: {stderr_tail or exit code N}"
      属性: .stderr_tail  ← 最后 8192 字节的 stderr
```

`tail_text(text, *, max_bytes=8192)` 工具函数：将长文本截取末尾 `max_bytes` 字节（UTF-8 边界安全），避免在 HTTP 响应中暴露过长的错误内容。

### 8.3 dispatch.py — 一次性派发

`spawn_worker(alias, worker_args) → SpawnResult`

通过 `importlib.util` 动态加载 `workers/run_worker.py`（`@lru_cache` 缓存），调用其 `spawn_worker()` 运行子进程并等待结束。

```python
SpawnResult(alias, returncode, duration_ms)
```

适用场景：`extract`、`download` 等一次性 CLI 调用。**当前业务代码不使用此路径**（生产使用 Session 模式）；主要用于测试和手动调试。

### 8.4 session.py — 长驻 Session

**这是生产路径**，对应 `serve` 子命令。

#### 全局单例注册表

```python
_sessions: dict[str, PersistentWorkerSession] = {}

def get_worker_session(alias: str) -> PersistentWorkerSession:
    # 线程安全，每个 alias 只创建一个实例
```

进程生命周期内，每个 alias 对应唯一一个 `PersistentWorkerSession` 实例。

#### PersistentWorkerSession 公开接口

| 方法 | 说明 |
|------|------|
| `start() → SessionStartResult` | 启动子进程并等待 ready 信号；已在运行则返回 `newly_started=False` |
| `stop() → bool` | 发送 shutdown 命令，优雅退出；若超时则 terminate/kill；返回是否实际停止了进程 |
| `is_running() → bool` | 检查子进程是否仍在运行 |
| `ensure_started() → SessionStartResult` | 同 `start()`，语义上强调"确保启动" |
| `extract_npy(input_path, output_path)` | 发送 extract 请求；进程已死则自动重启再发 |

#### 内部实现要点

**`_start_locked()`：**
1. `subprocess.Popen(["uv","run",cli,"serve"], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True, bufsize=1)`
2. 启动 `_StderrDrainer` 后台线程（防死锁）
3. 调用 `_read_json_response(proc.stdout, 300s)` 等待 `{"ok":true,"event":"ready"}`
4. 检查进程是否已退出（`proc.poll() is not None`）
5. 检查 `ready.get("ok")` 是否为 `True`
6. 全部通过后，保存 `self._proc = proc`

**`_request_locked(payload, timeout_sec)`：**
1. 检查进程是否存活
2. `_write_request(proc, payload)` → `stdin.write(json + "\n"); stdin.flush()`
3. `_read_json_response(proc.stdout, timeout_sec)` → 带超时的 readline
4. 检查进程是否在响应期间退出
5. 检查 `response.get("ok")`

**`_readline_with_timeout(stream, timeout_sec)`：**

Python 的 `stream.readline()` 本身不支持超时，因此通过启动一个守护线程 + `thread.join(timeout_sec)` 实现：

```python
box: list[str] = []
thread = threading.Thread(target=lambda: box.append(stream.readline()))
thread.start()
thread.join(timeout_sec)
if thread.is_alive():
    raise TimeoutError(...)
```

**`_shutdown_locked()`：**

```
1. 向 stdin 写 {"cmd": "shutdown"}
2. 等待 stdout 的确认行（最多 30 秒）
3. terminate() → 等待 5 秒 → kill()（如有必要）
```

**`_StderrDrainer`：**

后台线程持续 `for line in proc.stderr`，将每行追加到 `self._lines`（有锁保护）并写入 `logger.debug`。错误时通过 `tail()` 返回最后 8192 字节。

#### Session 状态图

```
               ┌──────────────────────────────────────────┐
               │                                          │
         start()                                    stop()
               │                                          │
    ┌──────────▼──────────┐                   ┌───────────▼───────────┐
    │       STARTING       │                  │        STOPPING        │
    │   Popen + 等待 ready  │                  │  shutdown + terminate  │
    └──────────┬──────────┘                   └───────────┬───────────┘
               │ ok                                       │
               ▼                                          ▼
    ┌──────────────────────┐                   ┌──────────────────────┐
    │       RUNNING        │◄── extract_npy()  │       STOPPED        │
    │   _proc 存活          │───►─────────────  │   _proc is None      │
    └──────────────────────┘                   └──────────────────────┘
               │ 进程崩溃                                  ▲
               └────────── extract_npy 自动重启 ───────────┘
```

---

## 9. Service 层与 HTTP 层

### Service 层（service/features/<worker>.py）

每个 worker 对应一个 service 模块，职责：

1. 持有 `asyncio.Lock`，保证同一时刻只有一个 `extract` 在运行
2. 接收 FastAPI `UploadFile`，读取字节
3. `asyncio.to_thread()` 调用同步的 `sync_extract_npy()`，不阻塞事件循环
4. `sync_extract_npy()` 写临时文件 → 调用 `session.extract_npy()` → 检查输出文件存在

```python
async with _features_lock:
    return await asyncio.to_thread(sync_extract_npy, file_bytes, filename)
```

`sync_extract_npy()` 的流程：
```
tempfile.mkdtemp() 创建 session_dir
    → 写 input.npy
    → validate_waveform_npy()  (校验 shape / dtype)
    → session.extract_npy(input_path, output_path)
    → 断言 output.npy 存在
    → 返回 (output_path, session_dir, download_name)
```

`session_dir` 由 `BackgroundTasks` 在响应发送后异步清理（`shutil.rmtree`）。

### HTTP 层（api/routes/features.py）

每个 worker 暴露三个端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/features/{alias}` | 上传 .npy，返回 feature .npy 文件 |
| `POST` | `/api/features/{alias}/start` | 预加载 worker（可选，不调用也会自动启动） |
| `POST` | `/api/features/{alias}/stop` | 停止 worker，释放 GPU/内存 |

---

## 10. 完整调用链（端到端）

以 `POST /api/features/chinese-hubert-base` 为例，完整追踪一次请求：

```
1. 客户端上传 waveform.npy
   HTTP POST /api/features/chinese-hubert-base
   Content-Type: multipart/form-data

2. extract_chinese_hubert_base_features() [routes/features.py]
   └─ 调用 extract_upload(file)

3. extract_upload() [service/features/chinese_hubert_base.py]
   ├─ await file.read()  ← 读取上传字节
   └─ async with _features_lock:
       └─ await asyncio.to_thread(sync_extract_npy, file_bytes, filename)

4. sync_extract_npy() [service/features/chinese_hubert_base.py]
   ├─ tempfile.mkdtemp() → /tmp/features-chinese-hubert-base-XXXX/
   ├─ 写 input.npy
   ├─ validate_waveform_npy(input_path)  ← 校验 shape / dtype
   └─ get_worker_session("chinese-hubert-base").extract_npy(input_path, output_path)

5. PersistentWorkerSession.extract_npy() [infra/worker/session.py]
   ├─ with self._lock:
   │   ├─ [首次或进程已死] _start_locked()
   │   │   ├─ build_worker_command("chinese-hubert-base", ["serve"])
   │   │   │   → ["uv","run","chinese-hubert-base","serve"]
   │   │   │     cwd = workers/huggingface/TencentGameMate/chinese-hubert-base/
   │   │   ├─ subprocess.Popen(...)  ← 启动子进程
   │   │   ├─ _StderrDrainer.start()
   │   │   └─ 等待 {"ok":true,"event":"ready"}（最多 300 秒）
   │   └─ _request_locked({"cmd":"extract","input":..., "output":...}, timeout_sec=120)
   │       ├─ proc.stdin.write('{"cmd":"extract",...}\n'); flush()
   │       └─ 等待 stdout 的 {"ok":true}（最多 120 秒）

6. Worker 子进程 (uv run chinese-hubert-base serve)
   ├─ serve.py: 读取 stdin 的 JSON Line
   ├─ _handle_request({"cmd":"extract", ...})
   ├─ extract.run_extract(input_path, output_path)
   │   ├─ load_waveform_npy(input_path)
   │   ├─ extract_features_single(waveform)
   │   │   ├─ get_runtime()  ← 使用缓存的 HubertModel
   │   │   └─ model(input_values).last_hidden_state
   │   └─ np.save(output_path, features)  → /tmp/.../output.npy
   └─ 向 stdout 写 {"ok":true}

7. 回到 sync_extract_npy()
   ├─ 断言 output_path.is_file()
   └─ 返回 (output_path, session_dir, "waveform_features.npy")

8. routes/features.py
   ├─ background_tasks.add_task(_cleanup_features_session, session_dir)
   └─ return FileResponse(output_path, filename="waveform_features.npy")

9. 客户端收到 feature .npy 文件
   Content-Type: application/octet-stream
   Content-Disposition: attachment; filename="waveform_features.npy"

10. [响应发送完毕后，后台任务] shutil.rmtree(session_dir)
```

---

## 11. 错误 → HTTP 状态码映射

| 异常 | 触发条件 | HTTP 状态码 | detail 内容 |
|------|----------|-------------|-------------|
| `WorkerSpawnTimeout` | 启动握手或推理超过 `timeout_sec` | **504 Gateway Timeout** | `"Worker '{alias}' timed out after Ns"` |
| `WorkerSpawnFailed` | 进程非零退出、stdout 关闭、响应 `ok=false` | **502 Bad Gateway** | stderr 最后 8192 字节 |
| `ValueError` | 上传文件非 ndarray、shape 不合法 | **422 Unprocessable Entity** | 校验错误描述 |
| `RuntimeError` | output.npy 不存在等内部错误 | **500 Internal Server Error** | 错误消息 |

---

## 12. 并发与线程安全

系统有两层锁，职责不同：

### asyncio.Lock（Service 层）

```python
_features_lock = asyncio.Lock()  # 每个 worker 一把

async with _features_lock:
    return await asyncio.to_thread(sync_extract_npy, ...)
```

- **作用域**：同一 alias 的 HTTP 请求串行化
- **原因**：每个 worker 进程是单线程服务循环，同时收到多个请求会导致协议混乱

### threading.Lock（Session 层）

```python
self._lock = threading.Lock()  # PersistentWorkerSession 内部

with self._lock:
    ...  # 所有 _start_locked / _request_locked / _shutdown_locked
```

- **作用域**：保护 `self._proc`、stdin/stdout 的读写
- **原因**：`asyncio.to_thread` 在线程池中运行，多个并发线程可能同时访问 Session

两层锁配合：HTTP 并发 → asyncio.Lock 串行 → 单线程调用 Session → threading.Lock 保护进程状态。

---

## 13. 平台差异（Linux vs Windows）

`run_worker.py` 的 `run_worker()` 函数（CLI 入口）根据 `os.name` 有不同行为：

### POSIX（Linux / macOS，生产环境）

使用 `os.execvp` 做**进程替换**：

```python
os.chdir(worker_dir)
os.execvp("uv", command)
```

`python run_worker.py` 这个 Python 进程被完全替换为 `uv run <cli>`，不存在「Python 套着 uv」的嵌套问题。FastAPI（通过 BackgroundTasks 或其他机制）直接监控 `uv` 进程。

### Windows（开发环境）

`os.execvp` 在 Windows 上行为不完整（不能真正替换进程），因此回退为 `subprocess.run`：

```python
result = subprocess.run(command, cwd=worker_dir, check=False)
return result.returncode
```

`PersistentWorkerSession._start_locked()` 始终使用 `subprocess.Popen`，不受此影响——上述区别仅在通过 `run_worker.py` **直接 CLI 调用** worker 时有效。

---

## 14. 已注册的 Workers

### 14.1 chinese-hubert-base

| 属性 | 值 |
|------|----|
| Alias | `chinese-hubert-base` |
| 模型来源 | HuggingFace：`TencentGameMate/chinese-hubert-base` |
| 目录 | `workers/huggingface/TencentGameMate/chinese-hubert-base/` |
| Python 版本 | 3.10.x |
| 主要依赖 | `torch`, `transformers`, `huggingface-hub`, `numpy` |
| 输入 | float32, 16 kHz, mono `.npy`，shape `(T,)` 或 `(1, T)` |
| 输出 | `last_hidden_state` `.npy`，shape `(1, T', 768)`，float32，BTC layout |
| 超时 | 120 秒 |
| 用途 | GPT-SoVITS fine-tuning pipeline A 的 HuBERT 特征提取 |

**注意**：此 worker 使用 GPT-SoVITS 特有的输入约定，跳过了官方 HuggingFace README 中的 `Wav2Vec2FeatureExtractor` 预处理步骤，直接喂 `input_values`。如需标准 HF 用法，需另行处理。

### 14.2 speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common

| 属性 | 值 |
|------|----|
| Alias | `speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common` |
| 模型来源 | ModelScope：`iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common` |
| 目录 | `workers/modelscope/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/` |
| Python 版本 | 3.10.x |
| 主要依赖 | `torch`, `torchaudio`, `modelscope[framework]`, `numpy` |
| 输入 | float32, 16 kHz, mono `.npy`，shape `(T,)` 或 `(1, T)`，振幅范围 `[-1, 1]` |
| 输出 | SV Embedding `.npy`，shape `(1, 20480)`，float32 |
| 超时 | 120 秒 |
| 用途 | GPT-SoVITS v2Pro pipeline B 的说话人向量提取 |

**注意**：振幅必须为常规 float32 范围（`[-1, 1]`），**不得**使用 HuBERT pipeline 的 1145.14 标度。推理内部先将波形转换为 80-bin Kaldi Fbank 特征，再过 ERes2NetV2 网络的 `forward3()` 接口。

模型权重下载到 worker 项目根下的 `.models/<models_dir>/`（见 `workers/registry.toml`），不在 `src/` 包内，以缩短路径并避免 Windows MAX_PATH 限制。

---

## 15. 本地开发与调试

### 手动下载模型

```bash
# Worker 目录内
cd workers/huggingface/TencentGameMate/chinese-hubert-base
uv run chinese-hubert-base download

cd workers/modelscope/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common
uv run speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common download
```

### 单次 CLI 推理（一次性，无长驻）

```bash
uv run chinese-hubert-base extract --input wave.npy --output feat.npy
uv run speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common extract --input wave.npy --output sv.npy
```

### 手动测试 serve 协议

```bash
# 启动 serve，手动输入 JSON
cd workers/huggingface/TencentGameMate/chinese-hubert-base
uv run chinese-hubert-base serve
# 进程输出: {"ok": true, "event": "ready"}
# 手动输入: {"cmd": "ping"}
# 进程输出: {"ok": true, "loaded": true}
# 手动输入: {"cmd": "shutdown"}
# 进程输出: {"ok": true, "event": "shutdown"}
```

### 运行 smoke test（SV worker）

```bash
cd workers/modelscope/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common
uv run python smoke_test.py
```

smoke_test.py 依次测试：内存推理、磁盘 CLI 推理、完整 serve 协议、通过 `run_worker.py` 调度。

### 通过 run_worker.py 调度

```bash
# 在 asr-server 项目根目录
uv run python workers/run_worker.py chinese-hubert-base -- extract \
    --input /tmp/wave.npy --output /tmp/feat.npy
```

### 运行单元测试

```bash
uv run pytest tests/test_worker_dispatch.py -v
```

测试覆盖：超时配置优先级、stderr tail 截断、`spawn_worker` 超时/失败路径、Session 并发 stderr 不死锁（35 次 extract）、HTTP 状态码映射（504/502）、`/start` / `/stop` 端点。

---

## 16. 如何新增一个 Worker

以新增 `my-new-model` 为例，按顺序完成以下步骤：

### 步骤 1：创建 worker uv 项目

```bash
mkdir -p workers/huggingface/MyOrg/my-new-model/src/my_new_model
cd workers/huggingface/MyOrg/my-new-model
```

创建 `pyproject.toml`：

```toml
[project]
name = "my-new-model"
version = "0.1.0"
description = "描述"
requires-python = ">=3.10,<3.11"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.26.0",
    # ... 其他依赖
]

[project.scripts]
my-new-model = "my_new_model.cli:main"

[build-system]
requires = ["uv_build>=0.11.14,<0.12.0"]
build-backend = "uv_build"
```

```bash
# 锁定依赖
uv lock
# 指定 Python 版本
echo "3.10" > .python-version
```

### 步骤 2：实现四个核心模块

参考现有 worker 结构，在 `src/my_new_model/` 下创建：

**`download.py`**
- `ensure_model_path() → tuple[Path, str]`：线程安全地解析或下载模型，返回 `(path, source)`
- `run_download()`：CLI 入口

**`extract.py`**
- `get_runtime()`：懒加载并缓存模型（线程安全，用 `threading.Lock`）
- `run_extract(input_path, output_path) → Path`：单次推理，npy → npy

**`serve.py`**：复制现有 `serve.py` 并修改 import，其结构几乎不需要改动。

**`cli.py`**：复制现有 `cli.py`，修改 `description` 和 `--input`/`--output` 的 `help` 文本。

### 步骤 3：在 registry.toml 中注册

```toml
[my-new-model]
source = "huggingface"
repo_id = "MyOrg/my-new-model"
path = "huggingface/MyOrg/my-new-model"
cli = "my-new-model"
description = "My new model description"
timeout_sec = 120
```

### 步骤 4：在 service 层添加模块

创建 `src/asr_server/service/features/my_new_model.py`，参考 `chinese_hubert_base.py`，修改：
- `WORKER_ALIAS = "my-new-model"`
- 输入/输出的校验逻辑和文件名格式

### 步骤 5：在 HTTP 路由层注册端点

在 `src/asr_server/api/routes/features.py` 中添加三个端点：
- `POST /api/features/my-new-model`
- `POST /api/features/my-new-model/start`
- `POST /api/features/my-new-model/stop`

### 步骤 6：验证

```bash
# 验证注册表解析
uv run python workers/run_worker.py my-new-model -- download

# 验证 serve 协议
cd workers/huggingface/MyOrg/my-new-model
uv run my-new-model serve

# 运行测试
uv run pytest tests/test_worker_dispatch.py -v
```
