import pytest
from unittest.mock import MagicMock, patch

def test_research_state_creation():
    from app.agents.graph import ResearchState
    state = {"query": "test query", "documents": [], "report": ""}
    assert "query" in state

def test_planner_agent_output():
    with patch("app.agents.crew.ResearchCrew") as mock_crew:
        mock_crew.return_value.run.return_value = {"plan": "Mocked plan"}
        crew = mock_crew()
        result = crew.run("test query")
        assert "plan" in result

def test_research_graph_nodes():
    with patch("app.agents.graph.ResearchGraph") as mock_graph:
        mock_graph.return_value.build_graph.return_value = True
        graph = mock_graph()
        assert graph.build_graph() is True
