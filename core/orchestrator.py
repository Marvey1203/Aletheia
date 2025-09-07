# core/orchestrator.py 

from typing import List, Dict
import re
from loguru import logger
from pydantic import BaseModel
from .llm import ModelManager

# --- Pydantic model for a clear, structured output ---

class CognitivePathway(BaseModel):
    """
    Defines the "cognitive team" and strategy for a given query.
    This is the output of the Conductor.
    """
    # FIX: Default models now match the roles in your models.json
    triage_model: str = "conductor_and_critic"
    plan_enrichment_model: str = "process_supervisor_4b"
    execution_model: str
    critique_model: str = "conductor_and_critic"
    notes: str

# --- The Conductor ---

class Conductor:
    """
    Analyzes a query and determines the optimal cognitive pathway.
    This is the "meta-mind" that orchestrates the orchestra.
    """
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
        # --- FIX: Routing rules now map to the roles in your models.json ---
        self.routing_rules: Dict[str, str] = {
            r"\b(code|script|function|python|javascript|rust)\b": "specialist_coder_8b",
            r"\b(story|poem|imagine|create|analyze|review)\b": "general_problem_solver_7b",
            r"\b(plan|steps|process|how to)\b": "general_problem_solver_7b",
        }
        logger.info("Conductor initialized with rule-based routing.")

    def determine_cognitive_pathway(self, query: str) -> CognitivePathway:
        """
        Applies a set of rules to the query to select the best models for the task.
        """
        logger.info("Conductor: Determining cognitive pathway...")
        query_lower = query.lower()
        
        for pattern, role in self.routing_rules.items():
            if re.search(pattern, query_lower):
                notes = f"Query matched pattern '{pattern}'. Routing to specialist: {role}."
                logger.info(notes)
                return CognitivePathway(
                    execution_model=role,
                    notes=notes
                )

        # FIX: Default execution model now matches a role in your models.json
        notes = "No specific rules matched. Defaulting to general problem solver."
        logger.info(notes)
        return CognitivePathway(
            execution_model="general_problem_solver_7b",
            notes=notes
        )

    def conduct_triage(self, query: str, gear_override: str = None) -> Dict:
        """
        A simplified, rule-based triage for determining reasoning depth.
        """
        logger.info("Conducting triage (rule-based).")
        if gear_override:
            return {"recommended_gear": gear_override, "reason": "User override."}
        if len(query.split()) < 5:
            return {"recommended_gear": "gear_1", "reason": "Short query."}
        else:
            return {"recommended_gear": "gear_2", "reason": "Default for moderate length query."}