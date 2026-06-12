from __future__ import annotations

import re

from rag.assembler.base import ContextChunk

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might must shall can i you he she it "
    "we they and or but if of in on at to for with by from".split()
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _content_words(text: str) -> frozenset[str]:
    return frozenset(w.lower().strip("\"'(),;:") for w in text.split() if w.lower() not in _STOP_WORDS)


class FaithfulnessChecker:
    """
    Heuristic faithfulness: fraction of answer sentences whose content words
    overlap sufficiently with the context.

    Lightweight approximation — replace with RAGAS or an NLI model in production.
    """

    def __init__(self, overlap_threshold: float = 0.5) -> None:
        self._threshold = overlap_threshold

    def score(self, answer: str, chunks: list[ContextChunk]) -> float:
        if not chunks:
            return 0.0
        context_words = _content_words(" ".join(c.text for c in chunks))
        sentences = _sentences(answer)
        if not sentences:
            return 1.0

        grounded = 0
        for sent in sentences:
            words = _content_words(sent)
            if not words:
                grounded += 1
                continue
            overlap = len(words & context_words) / len(words)
            if overlap >= self._threshold:
                grounded += 1

        return grounded / len(sentences)

    def is_faithful(self, answer: str, chunks: list[ContextChunk]) -> bool:
        return self.score(answer, chunks) >= self._threshold
