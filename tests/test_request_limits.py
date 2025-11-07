import sys
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import anita_slm_server.slm_server as slm


@pytest.fixture(scope="module")
def client():
    with TestClient(slm.app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _stub_backend(monkeypatch):
    class _DummyBackend:
        def generate(self, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
            return {
                "text": "ok",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            }

    monkeypatch.setattr(slm, "_init_backend_if_needed", lambda: None)
    slm.MODEL_STATE.backend = _DummyBackend()
    slm.MODEL_STATE.loaded_time = time.time()
    slm.MODEL_STATE.last_used_time = time.time()
    yield
    slm.MODEL_STATE.backend = None


def test_chat_max_tokens_hard_limit(client, monkeypatch):
    limit = slm.ENV["MAX_COMPLETION_TOKENS"]
    payload = {
        "model": "qwen2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": limit + 1,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "max_tokens" in response.json()["detail"]


def test_chat_message_count_limit(client, monkeypatch):
    monkeypatch.setitem(slm.ENV, "MAX_CHAT_MESSAGES", 2)
    payload = {
        "model": "qwen2.5-0.5b-instruct",
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    assert "messages" in response.json()["detail"]


def test_embeddings_input_count_limit(client, monkeypatch):
    monkeypatch.setitem(slm.ENV, "MAX_EMBEDDING_INPUTS", 2)
    payload = {
        "model": slm.DEFAULT_EMBEDDING_MODEL_ID,
        "input": ["a", "b", "c"],
    }
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    assert "input" in response.json()["detail"]
