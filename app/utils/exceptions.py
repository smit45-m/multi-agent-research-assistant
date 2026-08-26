"""
Custom exception classes for the application.
"""
from typing import Optional, Dict, Any


class ResearchAssistantError(Exception):
    """
    Base exception class for all Research Assistant errors.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message (str): The error message.
            details (Optional[Dict[str, Any]]): Additional error details.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AgentError(ResearchAssistantError):
    """Exception raised for agent execution failures."""
    pass


class RetrievalError(ResearchAssistantError):
    """Exception raised for RAG retrieval failures."""
    pass


class DocumentProcessingError(ResearchAssistantError):
    """Exception raised for document ingestion failures."""
    pass


class ConfigurationError(ResearchAssistantError):
    """Exception raised for configuration issues."""
    pass


class RateLimitError(ResearchAssistantError):
    """Exception raised for rate limiting issues."""
    pass
