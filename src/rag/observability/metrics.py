from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NoOpCounter:
    name: str

    def inc(self, amount: float = 1, labels: dict[str, str] | None = None) -> None:
        pass


@dataclass
class NoOpHistogram:
    name: str

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        pass


@dataclass
class NoOpGauge:
    name: str

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        pass


class _PrometheusMetrics:
    """Thin wrapper around prometheus_client if available, no-op otherwise."""

    def __init__(self) -> None:
        self._available = False
        try:
            import prometheus_client as prom  # type: ignore[import-not-found]
            self._prom = prom
            self._available = True
        except ImportError:
            pass
        self._counters: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}

    def counter(self, name: str, description: str = "", labels: list[str] | None = None) -> Any:
        if name not in self._counters:
            if self._available:
                self._counters[name] = self._prom.Counter(
                    name, description, labels or []
                )
            else:
                self._counters[name] = NoOpCounter(name)
        return self._counters[name]

    def histogram(self, name: str, description: str = "", labels: list[str] | None = None, buckets: list[float] | None = None) -> Any:
        if name not in self._histograms:
            if self._available:
                kwargs: dict[str, Any] = {}
                if buckets:
                    kwargs["buckets"] = buckets
                self._histograms[name] = self._prom.Histogram(
                    name, description, labels or [], **kwargs
                )
            else:
                self._histograms[name] = NoOpHistogram(name)
        return self._histograms[name]

    def gauge(self, name: str, description: str = "", labels: list[str] | None = None) -> Any:
        if name not in self._gauges:
            if self._available:
                self._gauges[name] = self._prom.Gauge(name, description, labels or [])
            else:
                self._gauges[name] = NoOpGauge(name)
        return self._gauges[name]


_metrics_instance: _PrometheusMetrics | None = None


def get_metrics() -> _PrometheusMetrics:
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = _PrometheusMetrics()
    return _metrics_instance
