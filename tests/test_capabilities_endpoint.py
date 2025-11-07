import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anita_slm_server import slm_server  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    return TestClient(slm_server.app)


def test_capabilities_endpoint_shape(client: TestClient):
    resp = client.get("/v1/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "main_model" in data and "embeddings" in data

    main = data["main_model"]
    assert isinstance(main.get("id"), str)
    assert isinstance(main.get("backend"), str)
    assert isinstance(main.get("loaded"), bool)
    assert isinstance(main.get("capabilities"), dict)
    # Ensure expected capability keys exist (even if False)
    for key in ["tools", "tool_choice", "max_input_tokens", "repetition_penalty", "stream"]:
        assert key in main["capabilities"]

    embeds = data["embeddings"]
    assert isinstance(embeds.get("models"), list)
    assert isinstance(embeds.get("capabilities"), dict)
    assert "encoding_format_float" in embeds["capabilities"]
    assert "encoding_format_base64" in embeds["capabilities"]
