"""Tests for worker timeout config, stderr tail, and dispatch error mapping."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
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
from asr_server.infra.worker.session import PersistentWorkerSession, get_worker_session

REPO_ROOT = Path(__file__).resolve().parent.parent
SV_ALIAS = "speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common"
SV_WORKER_DIR = (
    REPO_ROOT
    / "workers/modelscope/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common"
)
SV_TEST_INPUT = (
    SV_WORKER_DIR / ".local/test/manbo_slices_sv_inputs/manbo_0000_0000000000-0000214400.npy"
)
SV_MODEL_CKPT = (
    SV_WORKER_DIR
    / ".models/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common/pretrained_eres2netv2w24s4ep4.ckpt"
)

_FAKE_WORKER_SCRIPT = textwrap.dedent(
    """\
    import json
    import sys

    sys.stdout.write(json.dumps({"ok": True, "event": "ready"}) + "\\n")
    sys.stdout.flush()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        cmd = payload.get("cmd")
        if cmd == "shutdown":
            sys.stdout.write(json.dumps({"ok": True, "event": "shutdown"}) + "\\n")
            sys.stdout.flush()
            break
        if cmd == "extract":
            sys.stderr.write("Already present, skipping: " + ("x" * 200) + "\\n")
            sys.stderr.flush()
            sys.stdout.write(json.dumps({"ok": True}) + "\\n")
            sys.stdout.flush()
    """
)


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


class WorkerSessionStderrDrainTests(unittest.TestCase):
    ALIAS = "stderr-drain-test"

    def tearDown(self) -> None:
        get_worker_session(self.ALIAS).stop()

    def test_many_extracts_with_chatty_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            script_path = tmp_dir / "fake_worker.py"
            script_path.write_text(_FAKE_WORKER_SCRIPT, encoding="utf-8")
            input_path = tmp_dir / "input.npy"
            output_path = tmp_dir / "output.npy"
            np.save(input_path, np.zeros(1600, dtype=np.float32))

            def fake_build_worker_command(
                alias: str, worker_args: list[str]
            ) -> tuple[list[str], Path]:
                return [sys.executable, str(script_path), *worker_args], tmp_dir

            session = PersistentWorkerSession(self.ALIAS)
            with patch(
                "asr_server.infra.worker.session._load_run_worker_module"
            ) as load_mock:
                load_mock.return_value.build_worker_command = fake_build_worker_command
                with patch(
                    "asr_server.infra.worker.session.get_worker_timeout_sec",
                    return_value=5.0,
                ):
                    session.start()
                    for i in range(35):
                        with self.subTest(iteration=i + 1):
                            session.extract_npy(input_path, output_path)
                    session.stop()


class SvWorkerSessionIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        SV_TEST_INPUT.is_file() and SV_MODEL_CKPT.is_file(),
        "SV worker model or test input not available",
    )
    def test_many_extracts_without_timeout(self) -> None:
        session = get_worker_session(SV_ALIAS)
        session.stop()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / "input.npy"
            output_path = tmp_dir / "output.npy"
            input_path.write_bytes(SV_TEST_INPUT.read_bytes())

            session.start()
            try:
                for i in range(35):
                    with self.subTest(iteration=i + 1):
                        session.extract_npy(input_path, output_path)
                        out = np.load(output_path)
                        self.assertEqual(out.shape, (1, 20480))
                        self.assertEqual(out.dtype, np.float32)
            finally:
                session.stop()


class FeaturesRouteTests(unittest.TestCase):
    def tearDown(self) -> None:
        get_worker_session("chinese-hubert-base").stop()
        get_worker_session(SV_ALIAS).stop()

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


class SvFeaturesRouteTests(unittest.TestCase):
    SV_ALIAS = "speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common"
    SV_PATH = f"/api/features/{SV_ALIAS}"

    def tearDown(self) -> None:
        get_worker_session(self.SV_ALIAS).stop()

    def test_timeout_maps_to_504(self) -> None:
        client = TestClient(app)
        with patch(
            "asr_server.service.features.speech_eres2netv2w24s4ep4_sv_zh_cn_16k_common.get_worker_session"
        ) as session_factory:
            session = session_factory.return_value
            session.extract_npy.side_effect = WorkerSpawnTimeout(self.SV_ALIAS, 120)
            buf = io.BytesIO()
            np.save(buf, np.zeros(16000, dtype=np.float32))
            response = client.post(
                self.SV_PATH,
                files={"file": ("wave.npy", buf.getvalue(), "application/octet-stream")},
            )
        self.assertEqual(response.status_code, 504)

    def test_failed_spawn_maps_to_502(self) -> None:
        client = TestClient(app)
        with patch(
            "asr_server.service.features.speech_eres2netv2w24s4ep4_sv_zh_cn_16k_common.get_worker_session"
        ) as session_factory:
            session = session_factory.return_value
            session.extract_npy.side_effect = WorkerSpawnFailed(self.SV_ALIAS, 1, "boom")
            buf = io.BytesIO()
            np.save(buf, np.zeros(16000, dtype=np.float32))
            response = client.post(
                self.SV_PATH,
                files={"file": ("wave.npy", buf.getvalue(), "application/octet-stream")},
            )
        self.assertEqual(response.status_code, 502)
        self.assertIn("boom", response.json()["detail"])

    def test_start_stop_endpoints(self) -> None:
        client = TestClient(app)
        body_cls = __import__(
            "asr_server.schemas.features",
            fromlist=["FeaturesModelStateResponseBody"],
        ).FeaturesModelStateResponseBody

        with patch(
            "asr_server.service.features.speech_eres2netv2w24s4ep4_sv_zh_cn_16k_common.sync_start_session",
            return_value=body_cls(loaded=True, message="loaded"),
        ):
            start = client.post(f"{self.SV_PATH}/start")
        self.assertEqual(start.status_code, 200)
        self.assertTrue(start.json()["loaded"])

        with patch(
            "asr_server.service.features.speech_eres2netv2w24s4ep4_sv_zh_cn_16k_common.sync_stop_session",
            return_value=body_cls(loaded=False, message="released"),
        ):
            stop = client.post(f"{self.SV_PATH}/stop")
        self.assertEqual(stop.status_code, 200)
        self.assertFalse(stop.json()["loaded"])


if __name__ == "__main__":
    unittest.main()
