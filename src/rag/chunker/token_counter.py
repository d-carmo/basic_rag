from __future__ import annotations

from typing import Callable

# A tokenizer is any callable that maps a string to a token count.
Tokenizer = Callable[[str], int]


def word_count_tokenizer(text: str) -> int:
    """Word-based approximation: 1 word ≈ 1.33 tokens (reasonable for English)."""
    words = text.split()
    return max(1, int(len(words) * 1.33)) if words else 0


def make_tiktoken_tokenizer(encoding: str = "cl100k_base") -> Tokenizer:
    """Return a tiktoken-backed tokenizer. Raises ImportError if tiktoken is not installed."""
    import tiktoken  # type: ignore[import-not-found]

    enc = tiktoken.get_encoding(encoding)

    def _tokenize(text: str) -> int:
        return len(enc.encode(text))

    return _tokenize


def make_hf_tokenizer(model_name: str) -> Tokenizer:
    """Return a HuggingFace AutoTokenizer-backed tokenizer."""
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    tok = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return _tokenize
