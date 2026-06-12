from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
        return _NoOpSpan()

    @contextmanager
    def span(self, name: str, **kwargs: Any) -> Generator[_NoOpSpan, None, None]:
        yield _NoOpSpan()


class _OtelTracer:
    def __init__(self, service_name: str) -> None:
        from opentelemetry import trace  # type: ignore[import-not-found]
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import-not-found]
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # type: ignore[import-not-found]

        resource = Resource(attributes={SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(service_name)

    def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
        return self._tracer.start_as_current_span(name, **kwargs)

    @contextmanager
    def span(self, name: str, **kwargs: Any) -> Generator[Any, None, None]:
        with self._tracer.start_as_current_span(name, **kwargs) as s:
            yield s


_tracer_instance: Any = None


def get_tracer(service_name: str = "rag-pipeline") -> Any:
    global _tracer_instance
    if _tracer_instance is None:
        try:
            _tracer_instance = _OtelTracer(service_name)
        except ImportError:
            _tracer_instance = _NoOpTracer()
    return _tracer_instance
