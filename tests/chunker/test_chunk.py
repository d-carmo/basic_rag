from rag.chunker.chunk import BaseChunker, Chunk, ChunkMetadata
from rag.loaders.base import DocType, Document, DocumentMetadata


def _doc(text: str = "hello world") -> Document:
    return Document(text=text, metadata=DocumentMetadata())


def test_chunk_id_is_unique() -> None:
    meta = ChunkMetadata()
    c1 = Chunk(text="a", metadata=meta)
    c2 = Chunk(text="b", metadata=meta)
    assert c1.id != c2.id


def test_chunk_token_count_defaults_to_zero() -> None:
    c = Chunk(text="hello", metadata=ChunkMetadata())
    assert c.token_count == 0


def test_chunk_token_count_set_explicitly() -> None:
    c = Chunk(text="hello", token_count=5, metadata=ChunkMetadata())
    assert c.token_count == 5


def test_chunk_metadata_inherits_document_metadata() -> None:
    meta = ChunkMetadata(source_url="file.pdf", doc_type=DocType.PDF, chunk_index=3)
    assert meta.source_url == "file.pdf"
    assert meta.doc_type == DocType.PDF
    assert meta.chunk_index == 3


def test_chunk_metadata_defaults() -> None:
    meta = ChunkMetadata()
    assert meta.chunk_index == 0
    assert meta.parent_id is None


def test_chunk_metadata_parent_id() -> None:
    meta = ChunkMetadata(parent_id="abc-123")
    assert meta.parent_id == "abc-123"


def test_base_chunker_is_abstract() -> None:
    import inspect

    assert inspect.isabstract(BaseChunker)
