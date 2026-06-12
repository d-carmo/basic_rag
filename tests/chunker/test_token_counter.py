import pytest

from rag.chunker.token_counter import word_count_tokenizer


def test_word_count_positive() -> None:
    assert word_count_tokenizer("hello world foo") > 0


def test_empty_string_returns_zero() -> None:
    assert word_count_tokenizer("") == 0


def test_single_word() -> None:
    assert word_count_tokenizer("hello") >= 1


def test_proportional_to_length() -> None:
    short = word_count_tokenizer("one two")
    long = word_count_tokenizer("one two three four five six seven eight")
    assert long > short


def test_whitespace_only_returns_zero() -> None:
    assert word_count_tokenizer("   \n\t  ") == 0
