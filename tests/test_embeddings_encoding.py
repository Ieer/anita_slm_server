import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus

import numpy as np
import pytest
from fastapi.testclient import TestClient

import anita_slm_server.slm_server as slm


@pytest.fixture(autouse=True)
def _reset_embedding_state():
	slm.embedding_model_cache.clear()
	slm._embedding_last_used_times.clear()
	yield
	slm.embedding_model_cache.clear()
	slm._embedding_last_used_times.clear()


def _install_stub_backend(monkeypatch, encoder_factory):
	monkeypatch.setattr(slm, "SentenceTransformer", encoder_factory)
	monkeypatch.setattr(slm, "get_embedding_model_map", lambda force=False: {slm.DEFAULT_EMBEDDING_MODEL_ID: "stub-path"})
	monkeypatch.setattr(slm, "_resolve_embedding_device", lambda: "cpu")


def test_embeddings_returns_base64_when_requested(monkeypatch):
	class _StubEncoder:
		def __init__(self, *_args, **_kwargs):
			self.device = "cpu"

		def encode(self, inputs):
			return np.array([[0.125, -0.5, 1.5] for _ in inputs], dtype=np.float32)

		def to(self, device):
			self.device = device
			return self

	_install_stub_backend(monkeypatch, lambda *_a, **_k: _StubEncoder())
	client = TestClient(slm.app)
	payload = {
		"model": slm.DEFAULT_EMBEDDING_MODEL_ID,
		"input": ["hello"],
		"encoding_format": "base64",
	}
	resp = client.post("/v1/embeddings", json=payload)
	assert resp.status_code == HTTPStatus.OK
	embedding_str = resp.json()["data"][0]["embedding"]
	expected = base64.b64encode(np.array([0.125, -0.5, 1.5], dtype=np.float32).tobytes()).decode("ascii")
	assert embedding_str == expected


def test_embedding_model_load_is_singleton(monkeypatch):
	load_count = {"value": 0}

	class _SlowEncoder:
		def __init__(self, *_args, **_kwargs):
			load_count["value"] += 1
			time.sleep(0.05)
			self.device = "cpu"

		def encode(self, inputs):
			return np.zeros((len(inputs), 4), dtype=np.float32)

		def to(self, device):
			self.device = device
			return self

	_install_stub_backend(monkeypatch, lambda *_a, **_k: _SlowEncoder())
	req = slm.EmbeddingsRequest(model=slm.DEFAULT_EMBEDDING_MODEL_ID, input=["a"], encoding_format="float")
	start = threading.Barrier(3)

	def _invoke():
		start.wait()
		return slm._process_embedding_request(req, ["a"])

	with ThreadPoolExecutor(max_workers=2) as pool:
		futures = [pool.submit(_invoke) for _ in range(2)]
		start.wait()
		for fut in futures:
			fut.result()

	assert load_count["value"] == 1
