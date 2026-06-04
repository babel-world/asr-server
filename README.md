# ASR Server

A local, high-performance audio transcription API powered by faster-whisper and FastAPI.

## Tech Stack

![faster-whisper](https://img.shields.io/badge/faster--whisper-000000?style=flat&logo=openai&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-de5fe9?style=flat&logo=uv&logoColor=white)

## Prerequisites

Before running the server, ensure you have the following ready:

1. **GPU Execution (NVIDIA Libraries)**: Unlike `openai-whisper`, FFmpeg does not need to be installed on the system. For GPU execution (CUDA 12 / cuDNN 9), please refer to the official [faster-whisper requirements](https://github.com/SYSTRAN/faster-whisper#requirements) for detailed installation methods.
2. **uv**: Install the [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager (used for `uv sync` and `uv run` below).

## Local Deployment

1. Navigate to the project root directory (where `pyproject.toml` is located).
2. Sync the dependencies and start the local development server:

```bash
uv sync
uv run asr-server
```

The API will now be available locally (default: `http://127.0.0.1:19031`).

## Project Structure

```text
asr-server/
├── pyproject.toml
├── README.md
├── src/
│   └── asr_server/
│       ├── api/
│       │   ├── deps.py
│       │   ├── router.py
│       │   └── routes/
│       │       └── transcribe.py
│       ├── loaders/
│       │   └── cuda_utils.py
│       ├── main.py
│       ├── schemas/
│       │   └── transcribe.py
│       └── service/
│           ├── transcribe.py
│           └── whisper_model.py
└── uv.lock
```
