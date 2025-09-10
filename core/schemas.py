# core/schemas.py (V2.2 - The Self-Aware Mind)

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import TypedDict
import numpy as np
from numpy.typing import NDArray

# --- Core Trace Schemas (Validated for completeness) ---
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
    quant: Optional[str] = None
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
    timestamp: str
    model_info: ModelInfo
    persona_id: Optional[str] = None
    sub_traces: Optional[List[Trace]] = None


# --- Schemas for the Omega Core V2 & Self-Evolution Engine ---

class PlanStep(BaseModel):
    """
    Represents a single, structured step in a conceptual plan.
    """
    action_type: NDArray[np.float32] = Field(description="The H_vector for the type of action.")
    action_subject: Optional[NDArray[np.float32]] = Field(default=None, description="The optional H_vector for the subject of the action.")
    
    class Config:
        arbitrary_types_allowed = True

# --- NEW: THE SELFMODEL - THE AI'S AUTOBIOGRAPHY ---
class SelfModel(BaseModel):
    """
    Aletheia's formal model of its own capabilities, failures, and strategies.
    This is a dynamic, growing database of self-knowledge.
    """
    capabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Maps skills (e.g., 'casual_greeting', 'meta_cognition') to a statistically-derived confidence score."
    )
    failure_modes: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps query types or situations to common failure patterns."
    )
    effective_strategies: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps query types to proven, effective strategies discovered through self-correction."
    )

class GraphState(TypedDict):
    """
    Represents the state of our cognitive graph, now fully equipped for self-evolution.
    """
    # Core state
    query: str
    context: str
    
    # Social context
    social_context: Optional[Dict[str, float]]

    # Cognitive assets
    pathway: Optional[Dict[str, Any]]
    conceptual_plan: Optional[List[PlanStep]]
    linguistic_plan: Optional[List[str]]
    candidate_answer: Optional[str]
    scores: Optional[Dict[str, float]]
    
    # Meta-cognition and Self-Evolution state
    revision_history: List[str]
    self_model: Optional[SelfModel] # The AI's autobiography