"""
CrewAI orchestration for the research workflow.
"""
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.rag.vector_store import VectorStoreManager
from app.tools.search_tool import web_search_tool
from app.tools.retrieval_tool import create_retrieval_tool
from app.chains.prompts import (
    PLANNER_SYSTEM_PROMPT, 
    RETRIEVER_SYSTEM_PROMPT, 
    ANALYZER_SYSTEM_PROMPT, 
    WRITER_SYSTEM_PROMPT
)

class ResearchCrew:
    """Manages the CrewAI setup and execution for research."""
    
    def __init__(self, vector_store: VectorStoreManager):
        """
        Initialize the ResearchCrew.
        
        Args:
            vector_store (VectorStoreManager): Vector store for document retrieval.
        """
        settings = get_settings()
        kwargs = {
            "model": settings.OPENAI_MODEL_NAME,
            "api_key": settings.OPENAI_API_KEY
        }
        if settings.OPENAI_BASE_URL:
            kwargs["base_url"] = settings.OPENAI_BASE_URL
        self.llm = ChatOpenAI(**kwargs)
        
        self.vector_store = vector_store
        self.retrieval_tool = create_retrieval_tool(vector_store)
        
        # Initialize Agents
        self.planner = Agent(
            role="Research Planner",
            goal="Decompose complex queries into actionable research plans.",
            backstory=PLANNER_SYSTEM_PROMPT,
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        self.retriever = Agent(
            role="Information Retriever",
            goal="Find the most relevant and accurate information from internal and external sources.",
            backstory=RETRIEVER_SYSTEM_PROMPT,
            tools=[self.retrieval_tool, web_search_tool],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        self.analyzer = Agent(
            role="Data Analyzer",
            goal="Synthesize findings, identify contradictions, and evaluate source reliability.",
            backstory=ANALYZER_SYSTEM_PROMPT,
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        self.writer = Agent(
            role="Report Writer",
            goal="Produce a well-structured, clear, and comprehensive research report.",
            backstory=WRITER_SYSTEM_PROMPT,
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def build_crew(self) -> Crew:
        """
        Build and return the Crew instance with tasks.
        
        Returns:
            Crew: Configured CrewAI instance.
        """
        plan_task = Task(
            description="Create a detailed research plan for the query: {query}",
            expected_output="A structured plan outlining objectives, sub-questions, and sources to check.",
            agent=self.planner
        )
        
        retrieve_task = Task(
            description="Execute the research plan by retrieving documents and searching the web.",
            expected_output="A collection of relevant snippets and source URLs.",
            agent=self.retriever
        )
        
        analyze_task = Task(
            description="Analyze the retrieved information, find key themes, and resolve contradictions.",
            expected_output="A synthesis of findings and source reliability scores.",
            agent=self.analyzer
        )
        
        write_task = Task(
            description="Write a final markdown report including an executive summary and detailed findings.",
            expected_output="A complete markdown document.",
            agent=self.writer
        )
        
        return Crew(
            agents=[self.planner, self.retriever, self.analyzer, self.writer],
            tasks=[plan_task, retrieve_task, analyze_task, write_task],
            process=Process.sequential,
            verbose=True
        )
        
    def run(self, query: str) -> dict:
        """
        Run the CrewAI workflow.
        
        Args:
            query (str): The research query.
            
        Returns:
            dict: The final report and metadata.
        """
        crew = self.build_crew()
        result = crew.kickoff(inputs={"query": query})
        
        # result is typically a string in older CrewAI versions or CrewOutput in newer.
        report_text = str(result)
        
        return {
            "final_report": report_text,
            "metadata": {
                "process": "CrewAI",
                "agents_used": 4
            }
        }
