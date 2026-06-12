from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_OPEN_PATHS = frozenset({"/health", "/ready", "/metrics", "/docs", "/openapi.json", "/redoc"})


class TokenBucket:
    """In-memory token bucket for per-key rate limiting."""

    def __init__(self, rpm: int) -> None:
        self._rpm = max(1, rpm)
        self._tokens: float = float(self._rpm)
        self._last = time.monotonic()

    def consume(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(self._rpm, self._tokens + elapsed * (self._rpm / 60.0))
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate X-API-Key header and enforce per-key rate limits."""

    def __init__(self, app, api_keys: frozenset[str], rate_limit_rpm: int = 60) -> None:
        super().__init__(app)
        self._keys = api_keys
        self._rpm = rate_limit_rpm
        self._buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(self._rpm)
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _OPEN_PATHS:
            return await call_next(request)

        key = request.headers.get("X-API-Key", "")
        if not key or key not in self._keys:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

        if not self._buckets[key].consume():
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)

        return await call_next(request)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Emit per-request Prometheus counters and latency histograms."""

    def __init__(self, app) -> None:
        super().__init__(app)
        try:
            import prometheus_client as prom

            self._requests = prom.Counter(
                "http_requests_total",
                "Total HTTP requests",
                ["method", "path", "status"],
            )
            self._latency = prom.Histogram(
                "http_request_duration_seconds",
                "HTTP request duration in seconds",
                ["method", "path"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            )
            self._enabled = True
        except Exception:
            self._enabled = False

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._enabled:
            return await call_next(request)
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start
        path = request.url.path
        self._requests.labels(
            method=request.method, path=path, status=str(response.status_code)
        ).inc()
        self._latency.labels(method=request.method, path=path).observe(duration)
        return response
