"""Gateway behaviour tests.

Focused on the things the gateway exists to do that vLLM does not: auth,
scale-from-zero warming, readiness reflecting the upstream, and not exposing
vLLM's non-inference endpoints. The passthrough itself is deliberately dumb and
needs little proving.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app

UPSTREAM = "http://model-server-test:8000"
KEY = "test-gateway-key"


@pytest.fixture(autouse=True)
def _settings():
    get_settings.cache_clear()

    def _override() -> Settings:
        return Settings(
            model_server_url=UPSTREAM,
            gateway_api_key=KEY,
            require_auth=True,
            upstream_read_timeout_s=5.0,
            upstream_connect_timeout_s=2.0,
        )

    app.dependency_overrides[get_settings] = _override
    get_settings.cache_clear()
    # lifespan reads get_settings() directly rather than through DI, so patch
    # the cached value too.
    import app.config as config_module

    config_module.get_settings.cache_clear()
    original = config_module.Settings
    config_module.Settings = lambda **kw: _override()  # type: ignore[assignment]
    yield
    config_module.Settings = original  # type: ignore[assignment]
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def _auth() -> dict[str, str]:
    return {"authorization": f"Bearer {KEY}"}


@respx.mock
def test_rejects_missing_credential():
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={})
    assert resp.status_code == 401


@respx.mock
def test_rejects_wrong_credential():
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions", json={}, headers={"authorization": "Bearer nope"}
        )
    assert resp.status_code == 401


@respx.mock
def test_unexposed_vllm_route_is_404():
    """vLLM exposes /tokenize and adapter-loading endpoints; we must not."""
    with TestClient(app) as client:
        resp = client.post("/v1/load_lora_adapter", json={}, headers=_auth())
    assert resp.status_code == 404


@respx.mock
def test_warming_when_upstream_unreachable():
    """Scale-from-zero must read as 'warming', not as a generic failure."""
    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("no route to host")
    )
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={}, headers=_auth())

    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "service_warming"
    # Retry-After is what lets the caller render a warming state with a sane
    # backoff instead of guessing.
    assert resp.headers["retry-after"] == "30"


@respx.mock
def test_readiness_false_while_upstream_down():
    respx.get(f"{UPSTREAM}/health").mock(side_effect=httpx.ConnectError("down"))
    with TestClient(app) as client:
        resp = client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json()["status"] == "warming"


@respx.mock
def test_readiness_true_when_upstream_healthy():
    respx.get(f"{UPSTREAM}/health").mock(return_value=httpx.Response(200))
    with TestClient(app) as client:
        resp = client.get("/readyz")
    assert resp.status_code == 200


@respx.mock
def test_liveness_does_not_depend_on_upstream():
    """A dead model server must not get the gateway killed and restarted."""
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200


@respx.mock
def test_proxies_and_strips_caller_credential():
    captured: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.headers)
        return httpx.Response(200, json={"id": "cmpl-1", "choices": []})

    respx.post(f"{UPSTREAM}/v1/chat/completions").mock(side_effect=_handler)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "default", "messages": []},
            headers=_auth(),
        )

    assert resp.status_code == 200
    # The credential authenticates the caller to us; it is meaningless upstream
    # and forwarding it would widen where the secret travels.
    assert "authorization" not in {k.lower() for k in captured}
