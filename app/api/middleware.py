import time
import uuid
from typing import Callable, Awaitable, Dict, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import get_settings

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to check API key if configured."""
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        settings = get_settings()
        if settings.API_KEY:
            if request.url.path.startswith("/health") or request.url.path.startswith("/docs") or request.url.path.startswith("/openapi.json"):
                return await call_next(request)
            
            api_key = request.headers.get("X-API-Key")
            if not api_key or api_key != settings.API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Unauthorized", "detail": "Invalid or missing API key"}
                )
        return await call_next(request)

def configure_cors(app):
    """Configures CORS for the FastAPI application."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter."""
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.ip_records: Dict[str, List[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        if client_ip in self.ip_records:
            self.ip_records[client_ip] = [ts for ts in self.ip_records[client_ip] if now - ts < 60]
        else:
            self.ip_records[client_ip] = []
            
        if len(self.ip_records[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"error": "Too Many Requests", "detail": "Rate limit exceeded"}
            )
            
        self.ip_records[client_ip].append(now)
        return await call_next(request)

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to attach a unique request ID."""
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
