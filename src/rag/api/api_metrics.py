from __future__ import annotations

from typing import Any

try:
    import prometheus_client as _prom

    chunks_ingested_counter: Any = _prom.Counter(
        "chunks_ingested_total",
        "Total chunks written to the vector store",
        ["route"],
    )
except Exception:
    class _NoOp:
        def labels(self, **_): return self
        def inc(self, amount=1): pass

    chunks_ingested_counter: Any = _NoOp()
