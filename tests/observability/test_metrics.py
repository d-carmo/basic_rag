from __future__ import annotations

import pytest

from rag.observability.metrics import NoOpCounter, NoOpGauge, NoOpHistogram, get_metrics


def test_get_metrics_singleton():
    m1 = get_metrics()
    m2 = get_metrics()
    assert m1 is m2


def test_noop_counter_inc():
    c = NoOpCounter("test_counter")
    c.inc()
    c.inc(5.0, labels={"env": "test"})


def test_noop_histogram_observe():
    h = NoOpHistogram("test_hist")
    h.observe(0.5)
    h.observe(1.2, labels={"route": "/v1/query"})


def test_noop_gauge_set():
    g = NoOpGauge("test_gauge")
    g.set(42.0)


def test_metrics_counter_cached():
    m = get_metrics()
    c1 = m.counter("req_total", "Total requests")
    c2 = m.counter("req_total")
    assert c1 is c2


def test_metrics_histogram_cached():
    m = get_metrics()
    h1 = m.histogram("latency", "Latency")
    h2 = m.histogram("latency")
    assert h1 is h2


def test_metrics_gauge_cached():
    m = get_metrics()
    g1 = m.gauge("active_conns", "Active")
    g2 = m.gauge("active_conns")
    assert g1 is g2
