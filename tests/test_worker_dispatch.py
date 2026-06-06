"""Tests for worker timeout config, stderr tail, and dispatch error mapping."""

from __future__ import annotations

import io
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from asr_server.main import app
from asr_server.infra.worker.config import (
    clear_worker_config_cache,
    get_default_worker_timeout_sec,
    get_worker_timeout_sec,
)
from asr_server.infra.worker.errors import (
    WorkerSpawnFailed,
    WorkerSpawnTimeout,
    tail_text,
)
from asr_server.infra.worker.dispatch import spawn_worker


class WorkerConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_worker_config_cache()

    def tearDown(self) -> None:
        clear_worker_config_cache()

    def test_default_timeout_is_120(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_default_worker_timeout_sec(), 120.0)

    def test_registry_timeout_overrides_global(self) -> None:
        with patch.dict(os.environ, {"WORKER_SPAWN_TIMEOUT_SEC": "999"}, clear=False):
            self.assertEqual(get_worker_timeout_sec("chinese-hubert-base"), 120.0)

    def test_env_alias_overrides_global_when_no_registry_field(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WORKER_SPAWN_TIMEOUT_SEC": "999",
                "WORKER_SPAWN_TIMEOUT_UNKNOWN_WORKER": "45",
            },
            clear=False,
        ):
            self.assertEqual(get_worker_timeout_sec("unknown-worker"), 45.0)


class WorkerErrorsTests(unittest.TestCase):
    def test_tail_text_keeps_suffix(self) -> None:
        long_text = "a" * 9000 + "TAIL"
        self.assertTrue(tail_text(long_text, max_bytes=100).endswith("TAIL"))


class WorkerDispatchTests(unittest.TestCase):
    def test_spawn_timeout_raises(self) -> None:
        with patch(
            "asr_server.infra.worker.dispatch._load_run_worker_module"
        ) as load_mock:
            load_mock.return_value.spawn_worker.side_effect = subprocess.TimeoutExpired(
                cmd=["uv"],
                timeout=120,
            )
            with self.assertRaises(WorkerSpawnTimeout):
                spawn_worker("chinese-hubert-base", ["extract"])

    def test_spawn_failed_raises_with_tail(self) -> None:
        stderr = "x" * 9000 + "CUDA OOM"
        completed = subprocess.CompletedProcess(
            args=["uv"],
            returncode=1,
            stdout="",
            stderr=stderr,
        )
        with patch(
            "asr_server.infra.worker.dispatch._load_run_worker_module"
        ) as load_mock:
            load_mock.return_value.spawn_worker.return_value = completed
            with self.assertRaises(WorkerSpawnFailed) as ctx:
                spawn_worker("chinese-hubert-base", ["extract"])
            self.assertIn("CUDA OOM", ctx.exception.stderr_tail)


class FeaturesRouteTests(unittest.TestCase):
    def tearDown(self) -> None:
        from asr_server.infra.worker.session import get_worker_session

        get_worker_session("chinese-hubert-base").stop()

    def test_timeout_maps_to_504(self) -> None:
        client = TestClient(app)
        with patch(
            "asr_server.service.features.chinese_hubert_base.get_worker_session"
        ) as session_factory:
            session = session_factory.return_value
            session.extract_npy.side_effect = WorkerSpawnTimeout(
                "chinese-hubert-base", 120
            )
            buf = io.BytesIO()
            np.save(buf, np.zeros(16000, dtype=np.float32))
            response = client.post(
                "/api/features/chinese-hubert-base",
                files={"file": ("wave.npy", buf.getvalue(), "application/octet-stream")},
            )
        self.assertEqual(response.status_code, 504)

    def test_failed_spawn_maps_to_502(self) -> None:
        client = TestClient(app)
        with patch(
            "asr_server.service.features.chinese_hubert_base.get_worker_session"
        ) as session_factory:
            session = session_factory.return_value
            session.extract_npy.side_effect = WorkerSpawnFailed(
                "chinese-hubert-base", 1, "boom"
            )
            buf = io.BytesIO()
            np.save(buf, np.zeros(16000, dtype=np.float32))
            response = client.post(
                "/api/features/chinese-hubert-base",
                files={"file": ("wave.npy", buf.getvalue(), "application/octet-stream")},
            )
        self.assertEqual(response.status_code, 502)
        self.assertIn("boom", response.json()["detail"])

    def test_start_stop_endpoints(self) -> None:
        client = TestClient(app)
        with patch(
            "asr_server.service.features.chinese_hubert_base.sync_start_session",
            return_value=__import__(
                "asr_server.schemas.features",
                fromlist=["FeaturesModelStateResponseBody"],
            ).FeaturesModelStateResponseBody(
                loaded=True,
                message="loaded",
            ),
        ):
            start = client.post("/api/features/chinese-hubert-base/start")
        self.assertEqual(start.status_code, 200)
        self.assertTrue(start.json()["loaded"])

        with patch(
            "asr_server.service.features.chinese_hubert_base.sync_stop_session",
            return_value=__import__(
                "asr_server.schemas.features",
                fromlist=["FeaturesModelStateResponseBody"],
            ).FeaturesModelStateResponseBody(
                loaded=False,
                message="released",
            ),
        ):
            stop = client.post("/api/features/chinese-hubert-base/stop")
        self.assertEqual(stop.status_code, 200)
        self.assertFalse(stop.json()["loaded"])


if __name__ == "__main__":
    unittest.main()
