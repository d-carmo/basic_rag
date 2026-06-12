from __future__ import annotations

import pytest

from rag.api.config import Settings


def test_defaults():
    s = Settings(api_key="dev-key")
    assert "dev-key" in s.api_keys
    assert s.rate_limit_rpm == 60
    assert s.environment == "development"


def test_from_env_keys(monkeypatch):
    monkeypatch.setenv("API_KEY", "key1, key2,key3")
    s = Settings.from_env()
    assert s.api_keys == frozenset({"key1", "key2", "key3"})


def test_from_env_cors(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.com, https://b.com")
    s = Settings.from_env()
    assert "https://a.com" in s.cors_origins_list
    assert "https://b.com" in s.cors_origins_list


def test_from_env_rate_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_RPM", "120")
    s = Settings.from_env()
    assert s.rate_limit_rpm == 120


def test_api_keys_comma_separated():
    s = Settings(api_key="alpha, beta , gamma")
    assert s.api_keys == frozenset({"alpha", "beta", "gamma"})
