from __future__ import annotations

import pytest

from rag.assembler.base import AssembledContext, ContextChunk
from rag.eval.dataset import GoldenDataset, GoldenSample
from rag.eval.runner import EvalReport, EvalRunner
from rag.generation.pipeline import GenerationResult


class _FakeRetriever:
    def __init__(self, ids: list[str]):
        self._ids = ids

    async def retrieve(self, query, **kw):
        class _R:
            pass
        results = []
        for sid in self._ids:
            r = _R()
            r.source_id = sid
            results.append(r)
        return results


class _FakeAssembler:
    def __init__(self, ids: list[str]):
        self._ids = ids

    async def assemble(self, results, **kw):
        chunks = [ContextChunk(text="ctx", source_id=sid, score=1.0, rank=i) for i, sid in enumerate(self._ids)]
        return AssembledContext(
            chunks=chunks,
            citation_map={},
            total_tokens=10,
            truncated=False,
        )


class _FakeGen:
    def __init__(self, answer: str):
        self._answer = answer

    async def generate(self, query, context):
        return GenerationResult(answer=self._answer)


@pytest.mark.asyncio
async def test_runner_no_components():
    sample = GoldenSample(query="q", expected_answer="a", relevant_source_ids=["s1"])
    ds = GoldenDataset(name="test", samples=[sample])
    runner = EvalRunner()
    report = await runner.run(ds)
    assert len(report.sample_results) == 1
    r = report.sample_results[0]
    assert r.answer == ""
    assert r.context_recall == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_runner_with_retriever():
    sample = GoldenSample(query="q", expected_answer="cat sat", relevant_source_ids=["s1"])
    ds = GoldenDataset(name="test", samples=[sample])
    runner = EvalRunner(
        retriever=_FakeRetriever(["s1", "s2"]),
        assembler=_FakeAssembler(["s1", "s2"]),
        generation_pipeline=_FakeGen("cat sat mat"),
    )
    report = await runner.run(ds)
    r = report.sample_results[0]
    assert r.context_recall == pytest.approx(1.0)
    assert r.answer_similarity > 0.0


@pytest.mark.asyncio
async def test_aggregate_computed():
    ds = GoldenDataset(
        name="t",
        samples=[
            GoldenSample(query="q1", expected_answer="a", relevant_source_ids=[]),
            GoldenSample(query="q2", expected_answer="b", relevant_source_ids=[]),
        ],
    )
    runner = EvalRunner()
    report = await runner.run(ds)
    assert "answer_similarity" in report.aggregate
    assert "context_recall" in report.aggregate


def test_report_save(tmp_path):
    report = EvalReport(dataset_name="t", sample_results=[])
    report.compute_aggregate()
    out = str(tmp_path / "report.json")
    report.save(out)
    import json, pathlib
    data = json.loads(pathlib.Path(out).read_text())
    assert data["dataset"] == "t"
    assert "aggregate" in data
