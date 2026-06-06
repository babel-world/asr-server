"""Worker spawn timeout resolution (global default + per-alias overrides)."""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from typing import Any

from asr_server.utils.whisper_assets import get_repo_root

DEFAULT_WORKER_TIMEOUT_SEC = 120.0


def get_default_worker_timeout_sec() -> float:
    raw = os.getenv("WORKER_SPAWN_TIMEOUT_SEC")
    if raw is None or raw.strip() == "":
        return DEFAULT_WORKER_TIMEOUT_SEC
    return float(raw)


def _alias_to_env_suffix(alias: str) -> str:
    return alias.replace("-", "_").upper()


def _registry_timeout_sec(entry: dict[str, Any]) -> float | None:
    if "timeout_sec" not in entry:
        return None
    value = entry["timeout_sec"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"registry timeout_sec must be a number, got {value!r}")
    return float(value)


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, dict[str, Any]]:
    registry_path = get_repo_root() / "workers" / "registry.toml"
    if not registry_path.is_file():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")
    with registry_path.open("rb") as file:
        return tomllib.load(file)


def clear_worker_config_cache() -> None:
    """Clear cached registry reads (for tests)."""
    _load_registry.cache_clear()


def get_worker_timeout_sec(alias: str) -> float:
    """Resolve spawn timeout: registry > env per alias > global default."""
    registry = _load_registry()
    if alias in registry:
        registry_timeout = _registry_timeout_sec(registry[alias])
        if registry_timeout is not None:
            return registry_timeout

    env_key = f"WORKER_SPAWN_TIMEOUT_{_alias_to_env_suffix(alias)}"
    env_value = os.getenv(env_key)
    if env_value is not None and env_value.strip() != "":
        return float(env_value)

    return get_default_worker_timeout_sec()
