from pydantic import BaseModel, Field
from typing import Literal, Optional, List

class ResearchQuery(BaseModel):
    """
    Request model for initiating a research task.
    """
    query: str = Field(..., min_length=3, description="The research question or topic.")
    depth: Literal['quick', 'standard', 'deep'] = Field('standard', description="Depth of the research.")
    max_sources: int = Field(10, ge=1, le=50, description="Maximum number of sources to retrieve.")
    source_filters: Optional[List[str]] = Field(default=None, description="Filters to apply on sources.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the latest advancements in solid-state batteries?",
                "depth": "standard",
                "max_sources": 5,
                "source_filters": ["arxiv", "nature"]
            }
        }
    }

class DocumentUploadMetadata(BaseModel):
    """
    Metadata for uploaded documents.
    """
    tags: List[str] = Field(default_factory=list, description="Tags associated with the document.")
    description: Optional[str] = Field(default=None, description="A brief description of the document.")
