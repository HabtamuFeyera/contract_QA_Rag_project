import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "LexiRAG" in data["service"]
    assert data["status"] == "online"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_add_documents_and_query():
    # Ingest text
    ingest_res = client.post(
        "/add-documents/",
        json=[
            {"content": "Section 8. Non-Compete: The Advisor shall not engage in competing businesses for 12 months."}
        ]
    )
    assert ingest_res.status_code == 200

    # Query
    query_res = client.post(
        "/query/",
        json={"question": "What is the non-compete period?"}
    )
    assert query_res.status_code == 200
    data = query_res.json()
    assert "answer" in data
    assert len(data["citations"]) > 0
