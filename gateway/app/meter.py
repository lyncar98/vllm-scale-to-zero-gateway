"""Per-served-model counters for the status view and the usage log. Content-free by design.

What is kept, per served model name, over a rolling window of hours:
requests, errors, input and output tokens (from the upstream's own ``usage`` field), and request latency
samples for a p95. Nothing about the prompt or the completion is kept or logged.

Counters live in this process. The gateway runs as several replicas, so each one reports its own window, and
the status response says so.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

_MAX_SAMPLES_PER_HOUR = 1000
_MAX_SNIFF_BYTES = 1024 * 1024
_MAX_LINE_BYTES = 64 * 1024


@dataclass
class _Hour:
    requests: int = 0
    errors: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latencies_ms: list[float] = field(default_factory=list)


class Window:
    def __init__(self, hours: int) -> None:
        self.hours = hours
        self._hours: dict[int, _Hour] = {}

    def record(self, *, at: float, status: int, latency_ms: float, tokens_in: int, tokens_out: int) -> None:
        index = int(at // 3600)
        hour = self._hours.setdefault(index, _Hour())
        hour.requests += 1
        if status >= 500:
            hour.errors += 1
        hour.tokens_in += tokens_in
        hour.tokens_out += tokens_out
        if len(hour.latencies_ms) < _MAX_SAMPLES_PER_HOUR:
            hour.latencies_ms.append(latency_ms)
        self._prune(at)

    def summary(self, *, now: float) -> dict:
        first = int(now // 3600) - self.hours + 1
        live = [h for i, h in self._hours.items() if i >= first]
        samples = sorted(x for h in live for x in h.latencies_ms)
        return {
            "requests_24h": sum(h.requests for h in live),
            "errors_24h": sum(h.errors for h in live),
            "tokens_in_24h": sum(h.tokens_in for h in live),
            "tokens_out_24h": sum(h.tokens_out for h in live),
            "p95_latency_ms_24h": _p95(samples),
        }

    def _prune(self, at: float) -> None:
        cutoff = int(at // 3600) - self.hours + 1
        for index in [i for i in self._hours if i < cutoff]:
            del self._hours[index]


def _p95(sorted_samples: list[float]) -> float | None:
    """Nearest-rank 95th percentile, or None with no samples."""
    if not sorted_samples:
        return None
    rank = math.ceil(0.95 * len(sorted_samples))
    return round(sorted_samples[rank - 1], 1)


class Meter:
    def __init__(self, hours: int) -> None:
        self.hours = hours
        self._windows: dict[str, Window] = defaultdict(lambda: Window(self.hours))
        self._in_flight: dict[str, int] = defaultdict(int)

    def begin(self, name: str) -> None:
        self._in_flight[name] += 1

    def end(self, name: str) -> None:
        self._in_flight[name] = max(0, self._in_flight[name] - 1)

    def in_flight(self, name: str) -> int:
        return self._in_flight[name]

    def record(self, name: str, *, status: int, latency_ms: float, tokens_in: int = 0, tokens_out: int = 0,
               now: float | None = None) -> dict:
        at = time.time() if now is None else now
        self._windows[name].record(
            at=at, status=status, latency_ms=latency_ms, tokens_in=tokens_in, tokens_out=tokens_out,
        )
        event = {
            "event": "gateway.usage",
            "served_model_name": name,
            "status": status,
            "latency_ms": round(latency_ms, 1),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
        return event

    def summary(self, name: str, *, now: float | None = None) -> dict:
        at = time.time() if now is None else now
        base = self._windows[name].summary(now=at) if name in self._windows else Window(self.hours).summary(now=at)
        return {**base, "in_flight": self._in_flight[name]}


class UsageSniffer:
    """Finds the upstream's ``usage`` object in a response without keeping the response.

    A plain JSON body is read whole, up to a cap. A server-sent-events stream is read a line at a time, and only
    the lines that mention ``usage`` are parsed. vLLM sends usage on the last chunk when the caller asks for it.
    """

    def __init__(self, *, sse: bool) -> None:
        self._sse = sse
        self._buffer = b""
        self._usage: dict | None = None
        self._overflow = False

    def feed(self, chunk: bytes) -> None:
        if self._sse:
            self._buffer += chunk
            while b"\n" in self._buffer:
                line, self._buffer = self._buffer.split(b"\n", 1)
                self._line(line)
            if len(self._buffer) > _MAX_LINE_BYTES:
                self._buffer = b""
        elif not self._overflow:
            self._buffer += chunk
            if len(self._buffer) > _MAX_SNIFF_BYTES:
                self._overflow = True
                self._buffer = b""

    def finish(self) -> tuple[int, int]:
        if self._sse:
            if self._buffer:
                self._line(self._buffer)
        elif not self._overflow and self._buffer:
            self._take(self._buffer)
        self._buffer = b""
        if not self._usage:
            return 0, 0
        return int(self._usage.get("prompt_tokens") or 0), int(self._usage.get("completion_tokens") or 0)

    def _line(self, line: bytes) -> None:
        if b'"usage"' not in line:
            return
        line = line.strip()
        if line.startswith(b"data:"):
            line = line[len(b"data:"):].strip()
        self._take(line)

    def _take(self, payload: bytes) -> None:
        try:
            doc = json.loads(payload)
        except ValueError:
            return
        usage = doc.get("usage") if isinstance(doc, dict) else None
        if isinstance(usage, dict):
            self._usage = usage


def probe_key(upstream: str) -> str:
    return upstream.rstrip("/")
