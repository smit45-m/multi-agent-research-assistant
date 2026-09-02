"""
Integration tests for the research API endpoints.
"""
import pytest
from unittest.mock import patch

def test_submit_research_query(test_client):
    response = test_client.post("/api/v1/research/sync", json={
        "query": "What are autonomous agents?",
        "depth": "standard",
        "max_sources": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "report" in data
    assert data["status"] == "completed"
    assert data["confidence_score"] >= 0.8
    assert data["response_accuracy_score"] >= 0.85

def test_submit_research_query_async(test_client):
    response = test_client.post("/api/v1/research/", json={
        "query": "What are autonomous agents in AI?",
        "depth": "standard",
        "max_sources": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] in ["pending", "completed"]

def test_submit_research_query_invalid(test_client):
    response = test_client.post("/api/v1/research/", json={
        "query": "",
        "depth": "standard",
        "max_sources": 5
    })
    assert response.status_code == 422

def test_submit_research_query_short(test_client):
    response = test_client.post("/api/v1/research/", json={
        "query": "ab",
        "depth": "standard",
        "max_sources": 5
    })
    assert response.status_code == 422

def test_get_research_result_not_found(test_client):
    response = test_client.get("/api/v1/research/nonexistent-id")
    assert response.status_code == 404

def test_get_benchmark_endpoint(test_client):
    response = test_client.get("/api/v1/research/benchmark?max_cases=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total_test_cases"] == 10
    assert data["pass_rate_percentage"] == 100.0
    assert data["average_accuracy_percentage"] >= 85.0
