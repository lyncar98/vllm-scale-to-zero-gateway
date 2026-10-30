"""Routing by the request's `model` field, per-model metering, and the read-only status view.

Upstreams are mocked with respx. Each test writes its own routes file, so the routing table is the only thing
that differs between cases.
"""

from __future__ import annotations

import json
import logging
import os

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import app.config as config_module
from app.config import Settings, get_settings
from app.main import app
from app.meter import Meter, UsageSniffer, Window
from app.routes import RouteFileError, parse_routes

UPSTREAM = "http://model-server:8000"
MODEL_X = "model-finetune-7-abcdef12"
MODEL_X_UPSTREAM = f"http://{MODEL_X}.inference.svc:8000"
KEY = "test-gateway-key"
CHAT = "/v1/chat/completions"


@pytest.fixture
def routes_file(tmp_path):
    path = tmp_path / "routes.json"
    path.write_text(json.dumps({"models": [{
        "served_model_name": MODEL_X, "upstream": MODEL_X_UPSTREAM,
        "registration_id": "reg-x",
    }]}), encoding="utf-8")
    return path


@pytest.fixture
def gateway(routes_file, monkeypatch):
    settings = Settings(
        model_server_url=UPSTREAM, gateway_api_key=KEY, require_auth=True,
        served_model_name="default", routes_file=str(routes_file), status_window_hours=24,
        upstream_read_timeout_s=5.0, upstream_connect_timeout_s=2.0,
    )
    get_settings.cache_clear()
    original = config_module.Settings
    config_module.Settings = lambda **kw: settings  # type: ignore[assignment]
    get_settings.cache_clear()
    yield settings
    config_module.Settings = original  # type: ignore[assignment]
    get_settings.cache_clear()
    app.dependency_overrides.clear()


def _auth():
    return {"authorization": f"Bearer {KEY}"}


def _body(model):
    return {"model": model, "messages": [{"role": "user", "content": "private prompt"}]}


def _completion(prompt_tokens=12, completion_tokens=5):
    return httpx.Response(200, json={
        "id": "c1", "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "private answer"}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    })


# -- routing ------------------------------------------------------------------

@respx.mock
def test_a_request_for_a_deployed_model_goes_to_its_upstream_with_the_body_unchanged(gateway):
    sent = {}

    def capture(request):
        sent["body"] = json.loads(request.content)
        return _completion()

    respx.post(f"{MODEL_X_UPSTREAM}{CHAT}").mock(side_effect=capture)
    respx.post(f"{UPSTREAM}{CHAT}").mock(return_value=_completion())
    with TestClient(app) as client:
        r = client.post(CHAT, json=_body(MODEL_X), headers=_auth())
    assert r.status_code == 200
    assert sent["body"]["model"] == MODEL_X


@respx.mock
def test_the_baked_alias_still_goes_to_the_baked_model_server(gateway):
    respx.post(f"{UPSTREAM}{CHAT}").mock(return_value=_completion())
    with TestClient(app) as client:
        assert client.post(CHAT, json=_body("default"), headers=_auth()).status_code == 200


@respx.mock
def test_a_request_without_a_model_goes_to_the_alias(gateway):
    route = respx.post(f"{UPSTREAM}{CHAT}").mock(return_value=_completion())
    with TestClient(app) as client:
        assert client.post(CHAT, json={"messages": []}, headers=_auth()).status_code == 200
    assert route.called


@respx.mock
def test_an_undeployed_model_is_404_and_reaches_no_upstream(gateway):
    upstream = respx.post(f"{UPSTREAM}{CHAT}").mock(return_value=_completion())
    with TestClient(app) as client:
        r = client.post(CHAT, json=_body("model-nope-00000000"), headers=_auth())
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "model_not_found"
    assert not upstream.called


def test_a_broken_routes_file_fails_only_non_alias_models(gateway, routes_file):
    routes_file.write_text("{not json", encoding="utf-8")
    os.utime(routes_file, (1, 1))  # force a re-read even if the mtime did not move
    with TestClient(app) as client:
        r = client.post(CHAT, json=_body(MODEL_X), headers=_auth())
    assert r.status_code == 503 and r.json()["error"]["type"] == "routing_unavailable"


@respx.mock
def test_a_regenerated_routes_file_takes_effect_without_a_restart(gateway, routes_file):
    respx.post(f"{UPSTREAM}{CHAT}").mock(return_value=_completion())
    routes_file.write_text(json.dumps({"models": []}), encoding="utf-8")
    os.utime(routes_file, (1, 1))
    with TestClient(app) as client:
        assert client.post(CHAT, json=_body(MODEL_X), headers=_auth()).status_code == 404
        routes_file.write_text(json.dumps({"models": [{
            "served_model_name": MODEL_X, "upstream": MODEL_X_UPSTREAM,
        }]}), encoding="utf-8")
        os.utime(routes_file, (2, 2))
        respx.post(f"{MODEL_X_UPSTREAM}{CHAT}").mock(return_value=_completion())
        assert client.post(CHAT, json=_body(MODEL_X), headers=_auth()).status_code == 200


def test_the_routes_document_is_validated():
    with pytest.raises(RouteFileError):
        parse_routes({"models": [{"served_model_name": "a", "upstream": "http://a"}] * 2})
    with pytest.raises(RouteFileError):
        parse_routes({"models": [{"served_model_name": "default", "upstream": "http://a"}]}, alias="default")
    with pytest.raises(RouteFileError):
        parse_routes({"models": [{"served_model_name": "a", "upstream": "ftp://a"}]})
    with pytest.raises(RouteFileError):
        parse_routes({"not": "a list"})


# -- metering ------------------------------------------------------------------

@respx.mock
def test_tokens_are_taken_from_the_upstream_usage_and_kept_per_model(gateway, routes_file):
    respx.post(f"{UPSTREAM}{CHAT}").mock(return_value=_completion(prompt_tokens=12, completion_tokens=5))
    respx.get(f"{UPSTREAM}/health").mock(return_value=httpx.Response(200))
    respx.get(f"{MODEL_X_UPSTREAM}/health").mock(return_value=httpx.Response(200))
    with TestClient(app) as client:
        client.post(CHAT, json=_body("default"), headers=_auth())
        status = client.get("/internal/status", headers=_auth()).json()
    alias = next(m for m in status["models"] if m["alias"])
    assert (alias["requests_24h"], alias["tokens_in_24h"], alias["tokens_out_24h"]) == (1, 12, 5)


@respx.mock
def test_a_streamed_response_is_metered_from_its_final_usage_line(gateway):
    stream = (
        b'data: {"choices":[{"delta":{"content":"private"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" answer"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n'
        b"data: [DONE]\n\n"
    )
    respx.post(f"{UPSTREAM}{CHAT}").mock(return_value=httpx.Response(
        200, content=stream, headers={"content-type": "text/event-stream"},
    ))
    respx.get(f"{UPSTREAM}/health").mock(return_value=httpx.Response(200))
    respx.get(f"{MODEL_X_UPSTREAM}/health").mock(return_value=httpx.Response(200))
    with TestClient(app) as client:
        r = client.post(CHAT, json={**_body("default"), "stream": True}, headers=_auth())
        assert b"private answer" in r.content or b"private" in r.content
        status = client.get("/internal/status", headers=_auth()).json()
    alias = next(m for m in status["models"] if m["alias"])
    assert (alias["tokens_in_24h"], alias["tokens_out_24h"]) == (7, 3)


@respx.mock
def test_the_usage_log_names_the_served_model_and_never_the_content(gateway, caplog):
    respx.post(f"{MODEL_X_UPSTREAM}{CHAT}").mock(return_value=_completion(prompt_tokens=9, completion_tokens=4))
    with caplog.at_level(logging.INFO, logger="inference.gateway"), TestClient(app) as client:
        client.post(CHAT, json=_body(MODEL_X), headers=_auth())
    events = [json.loads(r.getMessage()) for r in caplog.records if r.getMessage().startswith("{")]
    usage = [e for e in events if e["event"] == "gateway.usage"]
    assert usage and usage[0]["served_model_name"] == MODEL_X
    assert (usage[0]["tokens_in"], usage[0]["tokens_out"], usage[0]["status"]) == (9, 4, 200)
    assert all("private" not in r.getMessage() for r in caplog.records)


def test_window_drops_hours_older_than_the_window():
    window = Window(hours=2)
    window.record(at=0, status=200, latency_ms=10, tokens_in=1, tokens_out=1)
    window.record(at=3 * 3600, status=200, latency_ms=20, tokens_in=2, tokens_out=2)
    summary = window.summary(now=3 * 3600)
    assert summary["requests_24h"] == 1 and summary["tokens_in_24h"] == 2


def test_p95_is_nearest_rank_and_none_without_samples():
    window = Window(hours=24)
    assert window.summary(now=0)["p95_latency_ms_24h"] is None
    for ms in range(1, 101):
        window.record(at=0, status=200, latency_ms=ms, tokens_in=0, tokens_out=0)
    assert window.summary(now=0)["p95_latency_ms_24h"] == 95


def test_errors_count_5xx_only():
    meter = Meter(hours=24)
    meter.record("m", status=200, latency_ms=1, now=0)
    meter.record("m", status=503, latency_ms=1, now=0)
    meter.record("m", status=404, latency_ms=1, now=0)
    assert meter.summary("m", now=0)["errors_24h"] == 1


def test_the_sniffer_reads_plain_json_and_ignores_oversize_bodies():
    s = UsageSniffer(sse=False)
    s.feed(b'{"usage":{"prompt_tokens":4,"completion_tokens":2}}')
    assert s.finish() == (4, 2)
    big = UsageSniffer(sse=False)
    big.feed(b"x" * (2 * 1024 * 1024))
    assert big.finish() == (0, 0)


# -- the status view -------------------------------------------------------------

def test_the_status_view_needs_the_same_credential(gateway):
    with TestClient(app) as client:
        assert client.get("/internal/status").status_code == 401


@respx.mock
def test_the_status_view_lists_the_alias_and_each_routed_model(gateway):
    respx.get(f"{UPSTREAM}/health").mock(return_value=httpx.Response(200))
    respx.get(f"{MODEL_X_UPSTREAM}/health").mock(side_effect=httpx.ConnectError("no pod"))
    with TestClient(app) as client:
        body = client.get("/internal/status", headers=_auth()).json()
    names = [m["served_model_name"] for m in body["models"]]
    assert names == ["default", MODEL_X]
    by_name = {m["served_model_name"]: m for m in body["models"]}
    assert by_name["default"]["healthy"] is True and by_name["default"]["alias"] is True
    assert by_name[MODEL_X]["healthy"] is False and by_name[MODEL_X]["registration_id"] == "reg-x"
    for m in body["models"]:
        assert m["replicas"] is None and m["in_flight"] == 0
        assert {"requests_24h", "errors_24h", "tokens_in_24h", "tokens_out_24h", "p95_latency_ms_24h"} <= m.keys()
    assert body["replica"]["window_hours"] == 24


@respx.mock
def test_the_status_view_is_503_when_the_routes_file_is_broken(gateway, routes_file):
    routes_file.write_text("[]", encoding="utf-8")
    os.utime(routes_file, (3, 3))
    with TestClient(app) as client:
        r = client.get("/internal/status", headers=_auth())
    assert r.status_code == 503 and r.json()["error"]["type"] == "routing_unavailable"
