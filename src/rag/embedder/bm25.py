from __future__ import annotations

import asyncio
import math
from collections import Counter

from rag.embedder.base import BaseEmbedder, SparseVector


class BM25SparseEmbedder(BaseEmbedder):
    """BM25 Okapi sparse embedder that produces term-weighted sparse vectors for Qdrant."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    @property
    def model_id(self) -> str:
        return "bm25"

    def fit(self, corpus: list[str]) -> None:
        """Build vocabulary and IDF table from a document corpus."""
        tokenized = [text.lower().split() for text in corpus]
        n = len(tokenized)
        self._avgdl = sum(len(t) for t in tokenized) / max(n, 1)

        df: dict[str, int] = {}
        for tokens in tokenized:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        self._vocab = {t: i for i, t in enumerate(sorted(df))}
        self._idf = {
            t: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for t, freq in df.items()
        }

    def _vectorize(self, text: str) -> SparseVector:
        if not self._vocab:
            raise RuntimeError("BM25SparseEmbedder.fit() must be called before embed_sparse()")
        tokens = text.lower().split()
        doc_len = len(tokens)
        counts = Counter(tokens)

        indices: list[int] = []
        values: list[float] = []
        for token, tf in counts.items():
            if token not in self._vocab:
                continue
            idf = self._idf[token]
            numer = tf * (self._k1 + 1)
            denom = tf + self._k1 * (1 - self._b + self._b * doc_len / max(self._avgdl, 1))
            weight = idf * numer / denom
            if weight > 0:
                indices.append(self._vocab[token])
                values.append(float(weight))
        return SparseVector(indices=indices, values=values)

    def _vectorize_batch(self, texts: list[str]) -> list[SparseVector]:
        return [self._vectorize(t) for t in texts]

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []
        return await asyncio.to_thread(self._vectorize_batch, texts)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "BM25SparseEmbedder only supports embed_sparse(); use a dense embedder for embed()"
        )
