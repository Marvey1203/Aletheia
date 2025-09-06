# core/schemas.py
# This file contains the Pydantic models that define the strict data structures
# for the Aletheia project, ensuring data integrity and validation.

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]
    result: Optional[str] = None

class Attempt(BaseModel):
    id: int
    plan: List[str]
    confidence_score: Optional[float] = None # For Manifold V2
    tool_calls: Optional[List[ToolCall]] = Field(default_factory=list)
    candidate: str
    scores: Dict[str, float]
    total: float

class Best(BaseModel):
    attempt_id: int
    candidate: str
    total: float

class Reflection(BaseModel):
    what_worked: str
    what_failed: str
    next_adjustment: str

class Summary(BaseModel):
    answer: str
    reasoning: str
    next_action: str

class ModelInfo(BaseModel):
    name: str
    quant: str
    runtime: str
    temp: float
    model_config = ConfigDict(protected_namespaces=())

class Trace(BaseModel):
    trace_id: str
    query: str
    seed_id: str
    summary: Summary
    attempts: List[Attempt]
    best: Best
    reflection: Reflection
    artifacts: Optional[List[str]] = Field(default_factory=list)
    timestamp: str # Using string for ISO 8601 format
    model_info: ModelInfo