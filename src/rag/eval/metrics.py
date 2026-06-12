from __future__ import annotations

import re
from collections import Counter

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can in on at to for of and or but".split()
)


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS]


def _token_overlap_f1(prediction: str, reference: str) -> float:
    pred = Counter(_tokens(prediction))
    ref = Counter(_tokens(reference))
    if not pred or not ref:
        return 0.0
    common = sum((pred & ref).values())
    precision = common / sum(pred.values())
    recall = common / sum(ref.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def answer_similarity(prediction: str, reference: str) -> float:
    """Token F1 between predicted and reference answer."""
    return _token_overlap_f1(prediction, reference)


def context_recall(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Fraction of relevant documents that appear in the retrieved set."""
    if not relevant_ids:
        return 1.0
    retrieved_set = set(retrieved_ids)
    return sum(1 for sid in relevant_ids if sid in retrieved_set) / len(relevant_ids)


def context_precision(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Fraction of retrieved documents that are relevant."""
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    return sum(1 for sid in retrieved_ids if sid in relevant_set) / len(retrieved_ids)


def faithfulness(answer: str, context_texts: list[str]) -> float:
    """Word-overlap faithfulness: max sentence overlap vs context."""
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sentences:
        return 0.0
    combined_context = " ".join(context_texts)
    return sum(_token_overlap_f1(s, combined_context) for s in sentences) / len(sentences)
