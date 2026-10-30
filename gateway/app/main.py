"""A thin inference gateway for a vLLM model server that scales to zero.

vLLM already speaks the OpenAI-compatible API, so this does NOT re-implement
inference semantics. It owns exactly four things vLLM does not:

  1. Caller authentication. vLLM's server has no auth model worth relying on,
     and an unauthenticated endpoint in front of a GPU is a direct route to
     someone else's compute bill.
  2. Warming as a first-class state. The GPU pool is scale-to-zero, so the
     first request after idle arrives while no GPU node exists. A naive proxy
     surfaces that as a timeout; this surfaces 503 + Retry-After so the caller
     can show "warming up" instead of "error".
  3. Readiness that reflects the *upstream*, not this process. A gateway pod
     that is up while vLLM is still loading weights must not report ready, or
     the Service will route traffic into a 503.
  4. Usage telemetry stripped of content. Token counts come from the
     upstream's own usage field, recorded with the served model name. Prompts
     and completions are never logged by this process.

It also routes one gateway to many models by the request's `model` field
(app/routes.py), and reports a read-only status view per served model
(GET /internal/status). Everything else is a transparent, stream-preserving
passthrough.

This is a standalone reference implementation: the pattern behind a serving
gateway described in "Before Our First Token" (the ChatLPO / AWS Startups
Custom-Model Inference Program series). See the repo README for the article
link and for what's deliberately out of scope here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import Settings, get_settings
from app.meter import Meter, UsageSniffer
from app.routes import RouteFileError, RouteStore

logger = logging.getLogger("inference.gateway")

# Routes proxied verbatim to vLLM. An allow-list, not a catch-all: vLLM also
# exposes /tokenize, /detokenize and (depending on build) a /load_lora_adapter
# endpoint that can load arbitrary adapters from disk. None of those should be
# reachable from outside the cluster.
_PROXY_ROUTES = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/models",
}
# Routes whose JSON body names the model. /v1/models has no body and always goes to the baked alias.
_MODEL_ROUTES = {"/v1/chat/completions", "/v1/completions", "/v1/embeddings"}
_PROBE_TTL_SECONDS = 10.0

# Hop-by-hop headers must not be forwarded (RFC 7230 6.1). Forwarding
# content-length in particular corrupts responses when the body is re-chunked.
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "content-encoding",
    "host",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    if settings.require_auth and not settings.gateway_api_key:
        # Loud, and fatal. Starting with auth required but no key configured
        # would reject every request -- better to fail at boot where it shows
        # up as CrashLoopBackOff with a clear reason than to serve 401s that
        # look like a caller problem.
        raise RuntimeError(
            "GATEWAY_GATEWAY_API_KEY is empty while GATEWAY_REQUIRE_AUTH is true. "
            "Set the secret or set GATEWAY_REQUIRE_AUTH=false for a local run."
        )

    app.state.client = _make_client(settings.model_server_url, settings)
    app.state.clients = {}
    app.state.probes = {}
    app.state.routes = RouteStore(
        alias=settings.served_model_name, alias_upstream=settings.model_server_url, path=settings.routes_file,
    )
    app.state.meter = Meter(settings.status_window_hours)
    try:
        yield
    finally:
        for client in [app.state.client, *app.state.clients.values()]:
            await client.aclose()


def _make_client(base_url: str, settings: Settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(
            connect=settings.upstream_connect_timeout_s,
            read=settings.upstream_read_timeout_s,
            write=settings.upstream_read_timeout_s,
            pool=settings.upstream_connect_timeout_s,
        ),
    )


def _client_for(app: FastAPI, upstream: str) -> httpx.AsyncClient:
    """The baked alias reuses the startup client. Every other model server gets its own pooled client."""
    if upstream == app.state.routes.alias_upstream:
        return app.state.client
    if upstream not in app.state.clients:
        app.state.clients[upstream] = _make_client(upstream, get_settings())
    return app.state.clients[upstream]


def _requested_model(route: str, body: bytes) -> str | None:
    """The `model` field of a JSON request body, or None. A body that is not JSON is left to vLLM to reject."""
    if route not in _MODEL_ROUTES:
        return None
    try:
        doc = json.loads(body)
    except ValueError:
        return None
    if isinstance(doc, dict) and isinstance(doc.get("model"), str) and doc["model"]:
        return doc["model"]
    return None


def _log_usage(event: dict) -> None:
    # One line per request: served name, status, latency and token counts. No prompt, no completion.
    logger.info(json.dumps(event, separators=(",", ":")))


app = FastAPI(
    title="vLLM scale-to-zero gateway",
    description=(
        "Private inference entrypoint in front of a scale-to-zero vLLM "
        "deployment. Not directly internet-facing."
    ),
    lifespan=lifespan,
)


def require_caller(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Authenticate the calling *service*. Not an end user."""
    if not settings.require_auth:
        return

    import hmac

    # Keys that are configured. An unset secondary key is left out entirely.
    keys = [settings.gateway_api_key]
    if settings.gateway_secondary_api_key:
        keys.append(settings.gateway_secondary_api_key)
    # Constant-time comparison: a plain != leaks key material through timing.
    accepted = bool(authorization) and any(
        hmac.compare_digest(authorization, f"Bearer {key}") for key in keys if key
    )
    if not accepted:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing gateway credential.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    """Liveness: is this process itself alive. Does not touch the upstream."""
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
async def readyz(request: Request) -> Response:
    """Readiness: is the *model server* able to serve.

    Reports not-ready while vLLM is still loading weights. Without this the
    Service would route traffic to a gateway whose upstream is not up yet, and
    the caller would see a 503 that looks like a gateway fault.
    """
    try:
        resp = await request.app.state.client.get("/health", timeout=5.0)
        if resp.status_code == 200:
            return JSONResponse({"status": "ready"})
        return JSONResponse(
            {"status": "upstream-unhealthy", "upstream_status": resp.status_code},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except httpx.RequestError as exc:
        # Expected during a scale-from-zero cold start; not an error worth
        # alerting on, so logged at debug.
        logger.debug("readiness probe could not reach model server: %s", exc)
        return JSONResponse(
            {"status": "warming"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@app.get("/internal/status", dependencies=[Depends(require_caller)])
async def status_view(request: Request) -> Response:
    """Read-only, per served model: health, in-flight requests, and the last 24 hours of usage.

    Counters belong to this replica. The gateway runs several, so a caller that wants the total adds
    them, and `replicas` stays null until the gateway can read the cluster's replica counts.
    """
    settings = get_settings()
    try:
        routes = request.app.state.routes.all()
    except RouteFileError as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "routing_unavailable"}},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    now = time.time()
    healthy = await asyncio.gather(*(_healthy(request.app, r.upstream) for r in routes))
    models = []
    for route, ok in zip(routes, healthy, strict=True):
        models.append({
            "served_model_name": route.served_model_name,
            "alias": route.served_model_name == settings.served_model_name,
            "registration_id": route.registration_id or None,
            "upstream": route.upstream,
            "healthy": ok,
            "replicas": None,
            "in_flight": request.app.state.meter.in_flight(route.served_model_name),
            **request.app.state.meter.summary(route.served_model_name, now=now),
        })
    return JSONResponse({
        "replica": {
            "id": socket.gethostname(),
            "window_hours": settings.status_window_hours,
            "generated_at": datetime.fromtimestamp(now, UTC).isoformat(),
        },
        "models": models,
    })


async def _healthy(app: FastAPI, upstream: str) -> bool:
    """Whether the model server answers /health. Cached briefly, so status polling cannot hammer a GPU pod."""
    now = time.monotonic()
    cached = app.state.probes.get(upstream)
    if cached and now - cached[0] < _PROBE_TTL_SECONDS:
        return cached[1]
    try:
        resp = await _client_for(app, upstream).get("/health", timeout=2.0)
        ok = resp.status_code == 200
    except httpx.RequestError:
        ok = False
    app.state.probes[upstream] = (now, ok)
    return ok


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST"],
    dependencies=[Depends(require_caller)],
)
async def proxy(path: str, request: Request) -> Response:
    route = f"/v1/{path}"
    if route not in _PROXY_ROUTES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{route} is not exposed by this gateway.",
        )

    body = await request.body()
    name = _requested_model(route, body) or get_settings().served_model_name
    try:
        target = request.app.state.routes.lookup(name)
    except RouteFileError as exc:
        logger.error("routing table unusable: %s", exc)
        return JSONResponse(
            {"error": {"message": "The gateway's routing table is unavailable.", "type": "routing_unavailable"}},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    if target is None:
        return JSONResponse(
            {"error": {"message": f"Model {name!r} is not deployed on this gateway.", "type": "model_not_found"}},
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return await _forward(request, route, body, name, target.upstream)


async def _forward(request: Request, route: str, body: bytes, name: str, upstream: str) -> Response:
    client = _client_for(request.app, upstream)
    meter: Meter = request.app.state.meter
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    # The caller's credential authenticates it to *us*; it must not be forwarded
    # upstream, where it has no meaning.
    headers.pop("authorization", None)

    started = time.monotonic()
    meter.begin(name)
    upstream_req = client.build_request(
        request.method,
        route,
        content=body,
        headers=headers,
        params=request.query_params,
    )

    try:
        resp = await client.send(upstream_req, stream=True)
    except httpx.ConnectError as exc:
        # No GPU node yet, or vLLM not listening. This is the scale-from-zero
        # path and is expected, so answer with something actionable rather than
        # a bare 502: Retry-After tells the caller how long to show "warming".
        meter.end(name)
        _log_usage(meter.record(name, status=503, latency_ms=(time.monotonic() - started) * 1000))
        logger.info("model server unreachable, reporting warming: %s", exc)
        return JSONResponse(
            {
                "error": {
                    "message": (
                        "Model server is warming up. The inference pool scales "
                        "to zero when idle; retry shortly."
                    ),
                    "type": "service_warming",
                }
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "30"},
        )
    except httpx.ReadTimeout:
        meter.end(name)
        _log_usage(meter.record(name, status=504, latency_ms=(time.monotonic() - started) * 1000))
        logger.warning("model server read timeout on %s", route)
        return JSONResponse(
            {"error": {"message": "Model server timed out.", "type": "upstream_timeout"}},
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        )

    response_headers = {
        k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    sniff = UsageSniffer(sse="text/event-stream" in resp.headers.get("content-type", ""))

    async def stream():
        # Streaming is preserved rather than buffered: token-by-token delivery is
        # the whole point of an inference API, and buffering would also hold a
        # long completion entirely in this pod's memory. The usage object is
        # sniffed on the way through, and the chunks themselves are not kept.
        try:
            async for chunk in resp.aiter_raw():
                sniff.feed(chunk)
                yield chunk
        finally:
            await resp.aclose()
            meter.end(name)
            tokens_in, tokens_out = sniff.finish()
            _log_usage(meter.record(
                name, status=resp.status_code, latency_ms=(time.monotonic() - started) * 1000,
                tokens_in=tokens_in, tokens_out=tokens_out,
            ))

    return StreamingResponse(
        stream(),
        status_code=resp.status_code,
        headers=response_headers,
        media_type=resp.headers.get("content-type"),
    )
