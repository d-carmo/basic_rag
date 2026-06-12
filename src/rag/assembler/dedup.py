from __future__ import annotations

from rag.assembler.base import ContextChunk


def _shingles(text: str, k: int = 5) -> frozenset[str]:
    normalized = " ".join(text.lower().split())
    return frozenset(normalized[i : i + k] for i in range(max(0, len(normalized) - k + 1)))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


class NearDuplicateFilter:
    """Remove near-duplicate chunks using Jaccard (default) or cosine similarity."""

    def __init__(self, threshold: float = 0.85, method: str = "jaccard") -> None:
        if method not in ("jaccard", "cosine"):
            raise ValueError(f"Unknown dedup method: {method!r}. Use 'jaccard' or 'cosine'.")
        self._threshold = threshold
        self._method = method

    def filter(
        self,
        chunks: list[ContextChunk],
        vectors: list[list[float]] | None = None,
    ) -> list[ContextChunk]:
        if not chunks:
            return []

        kept_indices: list[int] = []

        if self._method == "cosine" and vectors is not None:
            kept_vecs: list[list[float]] = []
            for i, vec in enumerate(vectors):
                if all(_cosine(vec, kv) < self._threshold for kv in kept_vecs):
                    kept_indices.append(i)
                    kept_vecs.append(vec)
        else:
            kept_shingles: list[frozenset[str]] = []
            for i, chunk in enumerate(chunks):
                sh = _shingles(chunk.text)
                if all(_jaccard(sh, ks) < self._threshold for ks in kept_shingles):
                    kept_indices.append(i)
                    kept_shingles.append(sh)

        return [chunks[i] for i in kept_indices]
