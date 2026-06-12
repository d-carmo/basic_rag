from __future__ import annotations

import time

import pytest

from rag.api.middleware import TokenBucket


def test_token_bucket_allows_initial():
    bucket = TokenBucket(rpm=60)
    assert bucket.consume() is True


def test_token_bucket_exhausts():
    bucket = TokenBucket(rpm=1)
    bucket._tokens = 1.0
    assert bucket.consume() is True
    bucket._tokens = 0.0
    bucket._last = time.monotonic()
    assert bucket.consume() is False


def test_token_bucket_refills():
    bucket = TokenBucket(rpm=60)
    bucket._tokens = 0.0
    bucket._last = time.monotonic() - 10
    assert bucket.consume() is True


def test_token_bucket_rpm_zero_clamps():
    bucket = TokenBucket(rpm=0)
    assert bucket._rpm >= 1
