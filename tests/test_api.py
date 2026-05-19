import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.main import app

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def mock_models():
    """Mock models to avoid the 'Models are not loaded yet!' error."""
    with patch("app.dependencies.models", {"chat_model": "mock_model", "retriever": "mock_retriever"}):
        yield


def test_chat_endpoint():
    response = client.post("/chat/", json={"query": "Hello"})
    assert response.status_code == 200
    assert "response" in response.json()
