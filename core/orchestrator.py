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
        # No longer need regex rules. The LLM is our router.
        self._build_orchestra_menu()
        logger.info("Conductor V2 (LLM-driven) initialized.")

    def _build_orchestra_menu(self):
        """Builds the 'menu' of available reasoners from the model roster."""
        self.principal_reasoners = {}
        # Dynamically find all models that are not the conductor or supervisor
        for role, definition in self.model_manager.roster.items():
            if "conductor" not in role and "supervisor" not in role:
                self.principal_reasoners[role] = definition.notes
        
        menu_items = [f"- {role}: {notes}" for role, notes in self.principal_reasoners.items()]
        self.orchestra_menu = "\n".join(menu_items)
        logger.info(f"Conductor built orchestra menu with {len(self.principal_reasoners)} reasoners.")

    def _generate_routing_prompt(self, query: str) -> str:
        """Constructs the prompt for the conductor model to make a routing decision."""
        # V2 PROMPT: More context, clearer instructions, and "Chain of Thought"
        return f"""You are the Conductor, a master of cognitive routing. Your task is to analyze the user's query and determine its fundamental INTENT. Do not be distracted by single keywords.

            First, think step-by-step about the user's core goal. Is it technical and logical, or is it creative and philosophical?
            Second, review your available musicians and their specialities.
            Finally, choose the single best musician for the query's true intent.

            **Orchestra Roster:**
            {self.orchestra_menu}

            **User Query:** "{query}"

            **Chain of Thought Analysis:**
            1.  **Core Intent:** (Analyze the user's intent here - e.g., "The user is asking for a personal opinion and a philosophical reflection on its own nature.")
            2.  **Musician Analysis:** (Analyze which musician best fits the intent - e.g., "The 'general_problem_solver_7b' is best for open-ended, creative, and reasoning-based tasks.")
            3.  **Final Selection:** (State the final choice)

            **Selected Role:**"""

    def determine_cognitive_pathway(self, query: str) -> CognitivePathway:
        """
        Uses the conductor model to intelligently select the best execution model.
        """
        logger.info("Conductor V2: Determining cognitive pathway via LLM...")
        
        routing_prompt = self._generate_routing_prompt(query)
        
        # Use our fast conductor model to make the decision
        selected_role = self.model_manager.generate_text(
            "conductor_and_critic",
            routing_prompt,
            max_tokens=20 # Just need the role name
        ).strip()

        # --- Validation and Fallback ---
        # Clean up the model's output to get only the role name
        # It might sometimes respond with "specialist_coder_8b." or similar.
        clean_role = re.sub(r'[^a-zA-Z0-9_-]', '', selected_role.split('\n')[0]).strip()

        if clean_role in self.principal_reasoners:
            notes = f"Conductor selected '{clean_role}' based on semantic analysis of the query."
            logger.success(notes)
            return CognitivePathway(
                execution_model=clean_role,
                notes=notes
            )
        else:
            # If the model hallucinates a role or fails, fall back to the default
            default_role = "general_problem_solver_7b"
            notes = f"Warning: Conductor LLM returned an invalid role ('{selected_role}'). Defaulting to '{default_role}'."
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