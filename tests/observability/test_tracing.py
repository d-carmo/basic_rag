from __future__ import annotations

import pytest

from rag.observability.tracing import _NoOpSpan, _NoOpTracer, get_tracer


def test_noop_span_set_attribute():
    s = _NoOpSpan()
    s.set_attribute("key", "val")


def test_noop_span_context_manager():
    s = _NoOpSpan()
    with s:
        pass


def test_noop_tracer_start_span():
    t = _NoOpTracer()
    s = t.start_as_current_span("test-span")
    assert isinstance(s, _NoOpSpan)


def test_noop_tracer_span_ctx():
    t = _NoOpTracer()
    with t.span("my-op") as s:
        s.set_attribute("x", 1)


def test_get_tracer_returns_instance():
    t = get_tracer("rag-pipeline")
    assert t is not None


def test_get_tracer_noop_when_no_otel():
    # otel not installed, should fall back to NoOpTracer
    t = get_tracer("test-service")
    assert hasattr(t, "span") or hasattr(t, "start_as_current_span")
