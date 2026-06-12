from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rag.eval import metrics as m
from rag.eval.dataset import GoldenDataset, GoldenSample


@dataclass
class SampleResult:
    query: str
    answer: str
    retrieved_ids: list[str]
    answer_similarity: float
    context_recall: float
    context_precision: float
    faithfulness: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    dataset_name: str
    sample_results: list[SampleResult]
    aggregate: dict[str, float] = field(default_factory=dict)

    def compute_aggregate(self) -> None:
        if not self.sample_results:
            self.aggregate = {
                "answer_similarity": 0.0,
                "context_recall": 0.0,
                "context_precision": 0.0,
                "faithfulness": 0.0,
            }
            return
        n = len(self.sample_results)
        self.aggregate = {
            "answer_similarity": sum(r.answer_similarity for r in self.sample_results) / n,
            "context_recall": sum(r.context_recall for r in self.sample_results) / n,
            "context_precision": sum(r.context_precision for r in self.sample_results) / n,
            "faithfulness": sum(r.faithfulness for r in self.sample_results) / n,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "aggregate": self.aggregate,
            "samples": [
                {
                    "query": r.query,
                    "answer": r.answer,
                    "answer_similarity": r.answer_similarity,
                    "context_recall": r.context_recall,
                    "context_precision": r.context_precision,
                    "faithfulness": r.faithfulness,
                }
                for r in self.sample_results
            ],
        }

    def save(self, path: str) -> None:
        import pathlib
        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2))


class EvalRunner:
    """Runs evaluation against a retrieval+generation pipeline."""

    def __init__(
        self,
        *,
        retriever: Any | None = None,
        assembler: Any | None = None,
        generation_pipeline: Any | None = None,
    ) -> None:
        self._retriever = retriever
        self._assembler = assembler
        self._gen = generation_pipeline

    async def _run_sample(self, sample: GoldenSample) -> SampleResult:
        retrieved_ids: list[str] = []
        context_texts: list[str] = []
        answer = ""

        if self._retriever is not None:
            results = await self._retriever.retrieve(sample.query)
            retrieved_ids = [r.source_id for r in results]

            if self._assembler is not None:
                ctx = await self._assembler.assemble(results)
                context_texts = [c.text for c in ctx.chunks]
                retrieved_ids = [c.source_id for c in ctx.chunks]

                if self._gen is not None:
                    result = await self._gen.generate(sample.query, ctx)
                    answer = result.answer

        return SampleResult(
            query=sample.query,
            answer=answer,
            retrieved_ids=retrieved_ids,
            answer_similarity=m.answer_similarity(answer, sample.expected_answer),
            context_recall=m.context_recall(retrieved_ids, sample.relevant_source_ids),
            context_precision=m.context_precision(retrieved_ids, sample.relevant_source_ids),
            faithfulness=m.faithfulness(answer, context_texts),
        )

    async def run(self, dataset: GoldenDataset) -> EvalReport:
        results = []
        for sample in dataset.samples:
            results.append(await self._run_sample(sample))
        report = EvalReport(dataset_name=dataset.name, sample_results=results)
        report.compute_aggregate()
        return report
