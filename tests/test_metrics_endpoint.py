import math
import sys
from collections.abc import Callable
from pathlib import Path

import pytest  # pylint: disable=import-error
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anita_slm_server import slm_server  # pylint: disable=import-error,wrong-import-position


HTTP_OK = 200
EXPECTED_TOTAL_TOKENS = 200.0
EXPECTED_MAX_TOKENS_PER_SEC = 60.0


@pytest.fixture(autouse=True)
def reset_metrics_state():
    """Ensure each test starts with pristine metric counters."""
    with slm_server._metrics_lock:  # pylint: disable=protected-access
        for key in list(slm_server._metrics.keys()):  # pylint: disable=protected-access
            slm_server._metrics[key] = 0.0  # pylint: disable=protected-access
    slm_server.MODEL_STATE.loaded_time = None
    slm_server.MODEL_STATE.last_used_time = None
    yield
    with slm_server._metrics_lock:  # pylint: disable=protected-access
        for key in list(slm_server._metrics.keys()):  # pylint: disable=protected-access
            slm_server._metrics[key] = 0.0  # pylint: disable=protected-access
    slm_server.MODEL_STATE.loaded_time = None
    slm_server.MODEL_STATE.last_used_time = None


@pytest.fixture()
def client() -> TestClient:
    return TestClient(slm_server.app)


def _parse_metric(text: str, name: str) -> float | None:
    prefix = f"{name} "
    for line in text.splitlines():
        if line.startswith(prefix):
            try:
                return float(line[len(prefix) :])
            except ValueError:
                return None
    return None


def _has_metric_line(text: str, predicate: Callable[[str], bool]) -> bool:
    return any(predicate(line) for line in text.splitlines())


def test_metrics_reports_latency_throughput_and_gpu(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    gpu_snapshot = [
        {
            "index": 0,
            "name": 'Mock "GPU"',
            "total_bytes": 16 * 1024 ** 3,
            "free_bytes": 12 * 1024 ** 3,
            "allocated_bytes": 3 * 1024 ** 3,
            "reserved_bytes": 4 * 1024 ** 3,
        }
    ]
    monkeypatch.setattr(slm_server, "get_gpu_memory_status", lambda: gpu_snapshot)

    slm_server.MODEL_STATE.loaded_time = 1_700_000_000.0

    with slm_server._metrics_lock:  # pylint: disable=protected-access
        slm_server._metrics_observe_chat_latency(0.5)  # pylint: disable=protected-access
        slm_server._metrics_observe_chat_latency(1.5)  # pylint: disable=protected-access
        slm_server._metrics_observe_chat_tokens(120, 2.0)  # 60 tokens/sec
        slm_server._metrics_observe_chat_tokens(80, 1.6)   # 50 tokens/sec

    response = client.get("/metrics")
    assert response.status_code == HTTP_OK

    body = response.text

    avg_latency = _parse_metric(body, "agent_slm_chat_latency_avg_seconds")
    assert avg_latency is not None and math.isclose(avg_latency, 1.0, rel_tol=1e-6)

    avg_tps = _parse_metric(body, "agent_slm_chat_tokens_per_sec_avg")
    assert avg_tps is not None and math.isclose(avg_tps, 55.0, rel_tol=1e-6)

    # GPU metrics should include sanitized label
    expected_label = 'agent_slm_gpu_memory_total_bytes{index="0",name="Mock \'GPU\'"}'
    assert _has_metric_line(body, lambda line: line.startswith(expected_label))

    # Base counters should also reflect totals
    assert _parse_metric(body, "agent_slm_chat_tokens_total") == EXPECTED_TOTAL_TOKENS
    assert _parse_metric(body, "agent_slm_chat_tokens_per_sec_max") == EXPECTED_MAX_TOKENS_PER_SEC


def test_metrics_omits_gpu_fields_when_absent(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(slm_server, "get_gpu_memory_status", lambda: [])

    response = client.get("/metrics")
    assert response.status_code == HTTP_OK

    body = response.text
    assert not _has_metric_line(body, lambda line: line.startswith("agent_slm_gpu_memory_"))
    assert _parse_metric(body, "agent_slm_chat_latency_avg_seconds") is None
    assert _parse_metric(body, "agent_slm_chat_tokens_per_sec_avg") is None