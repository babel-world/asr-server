import os

from fastapi import FastAPI

from asr_server.api.router import api_router
from asr_server.utils.whisper_assets import get_repo_root

app = FastAPI()
app.include_router(api_router)


def _load_dotenv() -> None:
    """Load repo-root .env into os.environ (does not override existing vars)."""
    env_path = get_repo_root() / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@app.get("/")
def hello() -> dict[str, str]:
    return {"message": "Hello World"}


def run() -> None:
    import uvicorn

    _load_dotenv()
    uvicorn.run(
        "asr_server.main:app", host="127.0.0.1", port=19031, reload=True
    )
