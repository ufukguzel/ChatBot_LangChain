"""Pydantic request / response models shared by API routes."""
from typing import List
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8192, description="User message")
    model: str = Field("gpt-4o-mini", description="OpenAI model identifier")
    temperature: float = Field(0.4, ge=0.0, le=2.0, description="Sampling temperature")
    rag: bool = Field(False, description="Whether to inject RAG context")


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class SourceInfo(BaseModel):
    id: str
    source: str
    snippet: str


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    sources: List[SourceInfo] = []


class UploadResponse(BaseModel):
    ok: bool
    message: str


class SourceDetail(BaseModel):
    id: str
    source: str
    text: str


class SessionResponse(BaseModel):
    session_id: str
