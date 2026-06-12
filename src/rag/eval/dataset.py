from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class GoldenSample(BaseModel):
    query: str
    expected_answer: str
    relevant_source_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoldenDataset(BaseModel):
    name: str
    description: str = ""
    samples: list[GoldenSample] = Field(default_factory=list)

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str | None = None) -> GoldenDataset:
        p = Path(path)
        samples = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(GoldenSample.model_validate(json.loads(line)))
        return cls(name=name or p.stem, samples=samples)

    def to_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            for s in self.samples:
                f.write(s.model_dump_json() + "\n")
