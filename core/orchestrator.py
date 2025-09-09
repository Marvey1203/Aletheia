# core/orchestrator.py (V2 - The Sentient Conductor)

from typing import List, Dict, Any
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
    triage_model: str = "conductor_and_critic"
    plan_enrichment_model: str = "process_supervisor_4b"
    execution_model: str
    critique_model: str = "conductor_and_critic"
    notes: str

# --- The Conductor ---

class Conductor:
    """
    Analyzes a query using an LLM to determine the optimal cognitive pathway.
    This is the "meta-mind" that orchestrates the orchestra.
    """
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self._build_orchestra_menu()
        logger.info("Conductor V2.1 (Hardened) initialized.")

    def _build_orchestra_menu(self):
        """Builds the 'menu' of available reasoners from the model roster."""
        self.principal_reasoners = {}
        for role, definition in self.model_manager.roster.items():
            # This 'if' statement now has the crucial extra condition.
            if (definition.type == 'llm' and
                    "conductor" not in role and
                    "supervisor" not in role):
                self.principal_reasoners[role] = definition.notes
        
        menu_items = [f"- {role}: {notes}" for role, notes in self.principal_reasoners.items()]
        self.orchestra_menu = "\n".join(menu_items)
        logger.success(f"Conductor built a VALIDATED orchestra menu with {len(self.principal_reasoners)} reasoners.")

    # CHANGED: The prompt is now much more direct about the required output format.
    def _generate_routing_prompt(self, query: str) -> str:
        """Constructs the prompt for the conductor model to make a routing decision."""
        return f"""You are the Conductor, a master of cognitive routing. Your task is to analyze the user's query and choose the single best musician from the orchestra roster to handle it.

            **Orchestra Roster:**
            {self.orchestra_menu}

            **User Query:** "{query}"

            **Analysis:** First, think step-by-step about the user's core intent. Then, from the roster, select the single best role.
            
            **Your Final Answer MUST be ONLY the role name and nothing else.**
            For example: general_problem_solver_7b

            **Selected Role:**"""

    def determine_cognitive_pathway(self, query: str) -> CognitivePathway:
        """
        Uses the conductor model to intelligently select the best execution model.
        """
        logger.info("Conductor V2.1: Determining cognitive pathway via LLM...")
        
        routing_prompt = self._generate_routing_prompt(query)
        
        selected_role_output = self.model_manager.generate_text(
            "conductor_and_critic",
            routing_prompt,
            max_tokens=50 # Increased slightly to allow for some "thought" before the answer
        ).strip()

        # --- CHANGED: More robust parsing logic ---
        # Instead of just cleaning the string, we actively search for a valid role within it.
        # This makes the system resilient even if the model adds extra text.
        
        # Check for the best match in the model's output
        found_role = None
        for role_name in self.principal_reasoners.keys():
            if role_name in selected_role_output:
                found_role = role_name
                break # Stop at the first match

        if found_role:
            notes = f"Conductor selected '{found_role}' based on semantic analysis of the query."
            logger.success(notes)
            return CognitivePathway(
                execution_model=found_role,
                notes=notes
            )
        else:
            default_role = "general_problem_solver_7b"
            notes = f"Warning: Conductor LLM returned an ambiguous response ('{selected_role_output}'). Defaulting to '{default_role}'."
            logger.warning(notes)
            return CognitivePathway(
                execution_model=default_role,
                notes=notes
            )

    def conduct_triage(self, query: str, gear_override: str = None) -> Dict[str, Any]:
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