"""The Responses API (`/v1/responses`) is proxied, routed by model, and metered."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.main import app
from app.meter import UsageSniffer
from tests.test_routing_and_status import (  # noqa: F401  (gateway/routes_file are fixtures)
    KEY,
    MODEL_X,
    MODEL_X_UPSTREAM,
    UPSTREAM,
    _auth,
    gateway,
    routes_file,
)

RESPONSES = "/v1/responses"


def _response(input_tokens=11, output_tokens=6):
    return httpx.Response(200, json={
        "id": "resp_1", "object": "response", "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "private answer"}]}],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": 17},
    })


@pytest.mark.usefixtures("gateway")
@respx.mock
def test_responses_goes_to_the_alias_and_is_metered_from_input_and_output_tokens():
    sent = {}

    def capture(request):
        sent["body"] = json.loads(request.content)
        return _response()

    respx.post(f"{UPSTREAM}{RESPONSES}").mock(side_effect=capture)
    respx.get(f"{UPSTREAM}/health").mock(return_value=httpx.Response(200))
    respx.get(f"{MODEL_X_UPSTREAM}/health").mock(return_value=httpx.Response(200))
    with TestClient(app) as client:
        r = client.post(
            RESPONSES, json={"model": "default", "input": "private prompt", "store": False}, headers=_auth()
        )
        status = client.get("/internal/status", headers=_auth()).json()
    assert r.status_code == 200
    assert sent["body"]["store"] is False
    alias = next(m for m in status["models"] if m["alias"])
    assert (alias["tokens_in_24h"], alias["tokens_out_24h"]) == (11, 6)


@pytest.mark.usefixtures("gateway")
@respx.mock
def test_responses_is_routed_by_the_model_field():
    route = respx.post(f"{MODEL_X_UPSTREAM}{RESPONSES}").mock(return_value=_response())
    with TestClient(app) as client:
        r = client.post(RESPONSES, json={"model": MODEL_X, "input": "hi"}, headers=_auth())
    assert r.status_code == 200 and route.called


@pytest.mark.usefixtures("gateway")
def test_responses_still_needs_the_gateway_credential():
    with TestClient(app) as client:
        assert client.post(RESPONSES, json={"model": "default", "input": "hi"}).status_code == 401


@pytest.mark.usefixtures("gateway")
def test_other_response_paths_stay_unexposed():
    with TestClient(app) as client:
        r = client.get("/v1/responses/resp_1", headers={"authorization": f"Bearer {KEY}"})
    assert r.status_code == 404


@pytest.mark.usefixtures("gateway")
@respx.mock
def test_a_streamed_response_is_metered_from_the_completed_event():
    stream = (
        b'event: response.created\ndata: {"type":"response.created","response":{"usage":null}}\n\n'
        b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"private"}\n\n'
        b'event: response.completed\n'
        b'data: {"type":"response.completed","response":{"usage":{"input_tokens":8,"output_tokens":2}}}\n\n'
    )
    respx.post(f"{UPSTREAM}{RESPONSES}").mock(return_value=httpx.Response(
        200, content=stream, headers={"content-type": "text/event-stream"},
    ))
    respx.get(f"{UPSTREAM}/health").mock(return_value=httpx.Response(200))
    respx.get(f"{MODEL_X_UPSTREAM}/health").mock(return_value=httpx.Response(200))
    with TestClient(app) as client:
        client.post(RESPONSES, json={"model": "default", "input": "hi", "stream": True}, headers=_auth())
        status = client.get("/internal/status", headers=_auth()).json()
    alias = next(m for m in status["models"] if m["alias"])
    assert (alias["tokens_in_24h"], alias["tokens_out_24h"]) == (8, 2)


def test_the_sniffer_reads_both_usage_shapes():
    chat = UsageSniffer(sse=False)
    chat.feed(b'{"usage":{"prompt_tokens":4,"completion_tokens":2}}')
    assert chat.finish() == (4, 2)
    responses = UsageSniffer(sse=False)
    responses.feed(b'{"usage":{"input_tokens":5,"output_tokens":3}}')
    assert responses.finish() == (5, 3)
