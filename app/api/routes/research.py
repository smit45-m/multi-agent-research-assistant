import time
import uuid
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from app.api.schemas.requests import ResearchQuery
from app.api.schemas.responses import ResearchResponse, SourceInfo

router = APIRouter(prefix="/api/v1/research", tags=["Research"])

# In-memory store for task results
TASKS: Dict[str, ResearchResponse] = {}

def perform_research_task(task_id: str, query: str, app_state: Any):
    """Background task to run research."""
    try:
        start_time = time.time()
        graph = app_state.research_graph
        
        state = graph.run(query)
        
        processing_time = time.time() - start_time
        
        sources = []
        if state.get("sources_cited"):
            for s in state["sources_cited"]:
                sources.append(
                    SourceInfo(
                        title=s.get("title", "Unknown"),
                        url_or_path=s.get("url_or_path", ""),
                        relevance_score=s.get("relevance_score", 0.0),
                        source_type=s.get("source_type", "document")
                    )
                )
                
        TASKS[task_id] = ResearchResponse(
            task_id=task_id,
            status="completed",
            query=query,
            report=state.get("final_report", ""),
            sources=sources,
            confidence_score=state.get("confidence_score", 0.0),
            processing_time_seconds=processing_time,
            created_at=datetime.utcnow()
        )
    except Exception as e:
        TASKS[task_id] = ResearchResponse(
            task_id=task_id,
            status="failed",
            query=query,
            created_at=datetime.utcnow()
        )

@router.post("/", response_model=ResearchResponse)
async def create_research_task(
    query_obj: ResearchQuery,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Initiates a new research task."""
    task_id = str(uuid.uuid4())
    
    response = ResearchResponse(
        task_id=task_id,
        status="pending",
        query=query_obj.query,
        created_at=datetime.utcnow()
    )
    TASKS[task_id] = response
    
    background_tasks.add_task(perform_research_task, task_id, query_obj.query, request.app.state)
    
    return response

@router.get("/{task_id}", response_model=ResearchResponse)
async def get_research_task(task_id: str):
    """Retrieves the status and result of a research task."""
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
