from __future__ import annotations

import re

_DEFAULT_PATTERNS: list[str] = [
    r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?",
    r"disregard\s+(?:all\s+)?(?:previous|above|prior)",
    r"forget\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?",
    r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|another)",
    r"act\s+as\s+(?:if\s+you\s+(?:are|were))",
    r"new\s+system\s+prompt",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"pretend\s+(?:you\s+are|to\s+be)\s+",
    r"override\s+(?:your\s+)?(?:instructions?|rules?|constraints?)",
]


class PromptInjectionGuard:
    """Scan queries for known prompt-injection patterns."""

    def __init__(self, patterns: list[str] | None = None) -> None:
        raw = patterns if patterns is not None else _DEFAULT_PATTERNS
        self._compiled = [re.compile(p, re.IGNORECASE) for p in raw]

    def is_safe(self, query: str) -> bool:
        return not any(p.search(query) for p in self._compiled)

    def check(self, query: str) -> str:
        """Return the query unchanged if safe; raise ValueError if injection is detected."""
        if not self.is_safe(query):
            raise ValueError("Potential prompt injection detected in query.")
        return query
