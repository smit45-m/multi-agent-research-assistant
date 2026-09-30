"""Compatibility facade: report the actual bounded engine, never fake CrewAI metrics."""
from app.agents.graph import ResearchGraph


class ResearchCrew:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self._initialized = False

    def _setup_crew(self):
        return False

    def run(self, query, **options):
        state = ResearchGraph(self.vector_store).run(query, orchestrator="crewai", **options)
        return {**state, "sources": state["sources_cited"],
                "metadata": {"process": state["orchestrator"], "requested_process": "crewai"}}
