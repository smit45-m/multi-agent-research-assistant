import pytest
from unittest.mock import patch

@pytest.mark.asyncio
async def test_submit_research_query(test_client):
    with patch("app.main.ResearchGraph.run") as mock_run:
        mock_run.return_value = {"report": "Detailed research report on agents."}
        response = await test_client.post("/api/v1/research", json={
            "query": "What are autonomous agents?",
            "depth": "comprehensive",
            "max_sources": 5
        })
        assert response.status_code == 200
        assert "report" in response.json()

@pytest.mark.asyncio
async def test_submit_research_query_invalid(test_client):
    response = await test_client.post("/api/v1/research", json={
        "query": "",
        "depth": "comprehensive",
        "max_sources": 5
    })
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_submit_research_query_short(test_client):
    response = await test_client.post("/api/v1/research", json={
        "query": "ab",
        "depth": "comprehensive",
        "max_sources": 5
    })
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_get_research_result_not_found(test_client):
    response = await test_client.get("/api/v1/research/nonexistent-id")
    assert response.status_code == 404
