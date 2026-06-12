from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from rag.eval.dataset import GoldenDataset, GoldenSample


def test_golden_sample_defaults():
    s = GoldenSample(query="q", expected_answer="a")
    assert s.relevant_source_ids == []
    assert s.tags == []


def test_dataset_from_jsonl(tmp_path):
    p = tmp_path / "golden.jsonl"
    samples = [
        {"query": "q1", "expected_answer": "a1", "relevant_source_ids": ["s1"]},
        {"query": "q2", "expected_answer": "a2"},
    ]
    p.write_text("\n".join(json.dumps(s) for s in samples) + "\n")
    ds = GoldenDataset.from_jsonl(p)
    assert ds.name == "golden"
    assert len(ds.samples) == 2
    assert ds.samples[0].query == "q1"
    assert ds.samples[0].relevant_source_ids == ["s1"]


def test_dataset_round_trip(tmp_path):
    p = tmp_path / "data.jsonl"
    ds = GoldenDataset(
        name="test",
        samples=[GoldenSample(query="q", expected_answer="a", tags=["x"])],
    )
    ds.to_jsonl(p)
    loaded = GoldenDataset.from_jsonl(p, name="test")
    assert loaded.samples[0].tags == ["x"]


def test_dataset_empty_lines(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text('{"query":"q","expected_answer":"a"}\n\n')
    ds = GoldenDataset.from_jsonl(p)
    assert len(ds.samples) == 1
