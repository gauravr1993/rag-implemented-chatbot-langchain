# tests/__init__.py

from fastapi.testclient import TestClient
from src.main import app

test_client = TestClient(app)  # Reusable test client