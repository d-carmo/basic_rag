from __future__ import annotations

import pytest

from rag.api.app import create_app
from rag.api.config import Settings

TEST_API_KEY = "test-key"


@pytest.fixture()
def settings():
    return Settings(api_key=TEST_API_KEY)


@pytest.fixture()
def app(settings):
    return create_app(settings=settings)


@pytest.fixture()
def auth_headers():
    return {"X-API-Key": TEST_API_KEY}
