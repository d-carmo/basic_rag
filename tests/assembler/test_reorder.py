"""Tests for LostInMiddleReorder."""

from rag.assembler.base import ContextChunk
from rag.assembler.reorder import LostInMiddleReorder


def _chunk(rank: int) -> ContextChunk:
    return ContextChunk(text=f"chunk {rank}", source_id="src", score=float(10 - rank), rank=rank)


reorder = LostInMiddleReorder()


def test_reorder_empty() -> None:
    assert reorder.reorder([]) == []


def test_reorder_single_chunk() -> None:
    chunks = [_chunk(0)]
    result = reorder.reorder(chunks)
    assert len(result) == 1
    assert result[0].rank == 0


def test_reorder_two_chunks() -> None:
    chunks = [_chunk(0), _chunk(1)]
    result = reorder.reorder(chunks)
    assert result[0].rank == 0
    assert result[-1].rank == 1


def test_reorder_preserves_all_chunks() -> None:
    chunks = [_chunk(i) for i in range(6)]
    result = reorder.reorder(chunks)
    assert len(result) == 6
    assert {r.rank for r in result} == set(range(6))


def test_reorder_best_at_position_zero() -> None:
    # rank 0 is best (highest score); must land at index 0
    chunks = [_chunk(i) for i in range(5)]
    result = reorder.reorder(chunks)
    assert result[0].rank == 0


def test_reorder_second_best_at_last_position() -> None:
    # rank 1 is second best; must land at last position
    chunks = [_chunk(i) for i in range(5)]
    result = reorder.reorder(chunks)
    assert result[-1].rank == 1


def test_reorder_four_chunks_exact_positions() -> None:
    # [c0, c1, c2, c3] → [c0, c2, c3, c1]
    chunks = [_chunk(i) for i in range(4)]
    result = reorder.reorder(chunks)
    assert result[0].rank == 0
    assert result[1].rank == 2
    assert result[2].rank == 3
    assert result[3].rank == 1
