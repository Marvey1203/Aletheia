# core/schemas.py
# This file contains the Pydantic models that define the strict data structures
# for the Aletheia project, ensuring data integrity and validation.
# (Version 2.0 - Recursive Traces for Polymath Protocol)

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]
    result: Optional[str] = None

class Attempt(BaseModel):
    id: int
    plan: List[str]
    confidence_score: Optional[float] = None
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
    quant: Optional[str] = None # Optional for dummy mode
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
    # --- NEW: Fields for Polymath Protocol ---
    persona_id: Optional[str] = None # e.g., "Architect", "Critic", "Arbiter"
    sub_traces: Optional[List[Trace]] = None # The recursive part!

