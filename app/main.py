import logging
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from app.config import get_settings
from app.utils.logger import setup_logger
from app.utils.exceptions import ResearchAssistantError, AgentError, RetrievalError, DocumentProcessingError, RateLimitError
from app.api.schemas.responses import ErrorResponse
from app.api.middleware import APIKeyMiddleware, RateLimitMiddleware, RequestIDMiddleware, configure_cors
from app.api.routes import health, research, documents
from app.rag.vector_store import VectorStoreManager
from app.agents.graph import ResearchGraph

logger = setup_logger("multi_agent_api", logging.INFO)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Initializing application resources...")
    
    # Initialize vector store
    vector_store = VectorStoreManager()
    vector_store.initialize()
    app.state.vector_store = vector_store
    
    # Initialize research graph
    research_graph = ResearchGraph(vector_store)
    research_graph.build_graph()
    app.state.research_graph = research_graph
    
    logger.info("Application started successfully.")
    yield
    
    logger.info("Shutting down application resources...")
    # Save vector store on shutdown
    if hasattr(app.state, "vector_store"):
        app.state.vector_store.save()
    logger.info("Application stopped.")

def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    app = FastAPI(
        title="Multi-Agent AI Research Assistant API",
        description="REST API layer for the Multi-Agent Research Assistant.",
        version="1.0.0",
        docs_url="/docs",
        lifespan=lifespan
    )

    # Add middlewares (order matters)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.MAX_CONCURRENT_REQUESTS * 10)
    app.add_middleware(APIKeyMiddleware)
    configure_cors(app)
    
    # Include routers
    app.include_router(health.router)
    app.include_router(research.router)
    app.include_router(documents.router)

    # Serve static frontend files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Serve frontend index.html at root
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """Serve the frontend application."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "Multi-Agent Research Assistant API", "docs": "/docs"}

    # Exception Handlers
    @app.exception_handler(ResearchAssistantError)
    async def custom_exception_handler(request: Request, exc: ResearchAssistantError):
        status_code = 500
        if isinstance(exc, RateLimitError):
            status_code = 429
        elif isinstance(exc, DocumentProcessingError):
            status_code = 400
        
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=exc.__class__.__name__,
                detail=str(exc),
                status_code=status_code
            ).model_dump()
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal Server Error",
                detail="An unexpected error occurred.",
                status_code=500
            ).model_dump()
        )

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=True
    )
