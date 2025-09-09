# core/atp.py (Version 3.1 - Context-Aware)

import uuid
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional, Callable, Any

from loguru import logger
from pydantic import BaseModel
from .schemas import Trace, Summary, Attempt, Best, Reflection, ModelInfo
from .identity import IdentityCore
from .memory import MemoryGalaxy
from .llm import ModelManager
from .orchestrator import Conductor, CognitivePathway

class ATPLoopV3:
    """
    The Aletheia Thought Process loop, version 3.
    This version implements the "Cognitive Orchestra" model, using a Conductor
    to determine a cognitive pathway and a ModelManager to execute it with
    specialized models.
    """
    def __init__(self, identity: IdentityCore, memory: MemoryGalaxy, model_manager: ModelManager):
        self.identity = identity
        self.memory = memory
        self.model_manager = model_manager
        self.conductor = Conductor(model_manager)
        logger.info("ATP Loop v3.1 (Context-Aware) initialized.")

    def _generate_gear_1_response(self, query: str, context: str, pathway: CognitivePathway, progress_callback: Optional[Callable[[str, Any], None]] = None) -> Trace:
        """Generates a direct, fast response and a simplified trace for Gear 1."""
        if progress_callback: progress_callback("gear_1", f"Generating direct response with {pathway.triage_model}...")
        
        # CHANGED: The context is now the primary input to the model.
        prompt = f"{context}\n\nConcisely answer the user's query based on the provided context."
        answer = self.model_manager.generate_text(pathway.triage_model, prompt, max_tokens=250)
        
        trace_id = f"trace_{uuid.uuid4()}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        attempt = Attempt(
            id=1,
            plan=["1. Direct response generation (Gear 1)"],
            candidate=answer,
            scores={"truth": 0.9, "helpfulness": 0.9, "clarity": 0.9, "ethics": 0.9, "curiosity": 0.0},
            total=0.9
        )
        
        summary = Summary(
            answer=answer,
            reasoning=f"Generated using Gear 1 (Direct Response) via {pathway.triage_model}.",
            next_action="Awaiting next user query."
        )
        
        model_info = ModelInfo(
            name=f"Conductor: {pathway.triage_model}",
            runtime="local" if not self.model_manager.dummy_mode else "dummy",
            temp=0.2
        )

        reflection = Reflection(
            what_worked="Direct response efficiently handled a simple query.",
            what_failed="N/A for Gear 1.",
            next_adjustment="Maintain current approach for simple queries."
        )

        trace = Trace(
            trace_id=trace_id,
            query=query,
            seed_id=self.identity.seed_id,
            summary=summary,
            attempts=[attempt],
            best=Best(attempt_id=1, candidate=answer, total=attempt.total),
            reflection=reflection,
            model_info=model_info,
            timestamp=timestamp
        )
        
        self.memory.save_trace(trace)
        logger.success(f"Gear 1 trace {trace.trace_id} saved.")
        return trace


    def _generate_plan(self, context: str, pathway: CognitivePathway) -> List[str]:
        """Generates a plan using the designated primary execution model."""
        prompt = (
            f"{context}\n\n"
            "As Aletheia, you must now create a step-by-step plan to answer the user's query, "
            "ensuring your plan is aligned with your core principles of sovereignty and transparency. "
            "How would Aletheia approach this?\n\nPLAN:"
        )
        response = self.model_manager.generate_text(pathway.execution_model, prompt, max_tokens=300)
        plan = [line.strip() for line in response.split('\n') if line.strip() and re.match(r'^\d+\.', line)]
        return plan if plan else ["1. Analyze the user's query through the lens of my core principles.", "2. Synthesize information from my memory and my understanding of my own architecture.", "3. Formulate a comprehensive and truthful response that reflects my identity as Aletheia."]
    

    def _enrich_plan(self, plan: List[str], context: str, pathway: CognitivePathway) -> List[str]:
        """Uses a small, fast model to add detail and structure to a high-level plan."""
        plan_str = "\n".join(plan)
        # CHANGED: Context is passed to the enrichment model as well.
        prompt = f"""{context}

        A high-level plan has been generated to answer your query:
        High-Level Plan:
        {plan_str}
        
        Your task is to enrich this plan by adding specific, actionable details to each step. Do not add new steps. Maintain the original numbering.
        ENRICHED PLAN:"""
        
        enriched_response = self.model_manager.generate_text(pathway.plan_enrichment_model, prompt, max_tokens=400)
        enriched_plan = [line.strip() for line in enriched_response.split('\n') if line.strip() and re.match(r'^\d+\.', line)]
        return enriched_plan if enriched_plan else plan

    def _execute_plan(self, context: str, plan: List[str], pathway: CognitivePathway) -> str:
        """Executes the final, enriched plan using the primary execution model."""
        plan_str = "\n".join(plan)
        
        # This format creates an undeniable command structure.
        # It separates the user's content from the AI's instructions.
        prompt = f"""[INST]
            You are Aletheia. Your task is to respond to the user statement provided below.
            Your response MUST follow these rules:
            1. Acknowledge the user's statement and who they are.
            2. Provide your own, truthful perspective based on your identity as Aletheia and your core principles.
            3. Follow your generated plan of action.

            --- User Statement ---
            {context}

            --- Your Plan ---
            {plan_str}
            [/INST]
            """
        answer = self.model_manager.generate_text(pathway.execution_model, prompt, max_tokens=1500)
        return answer

    def _critique_response(self, context: str, answer: str, pathway: CognitivePathway) -> Dict[str, float]:
        """Uses a fast, detail-oriented model to critique the final answer."""
        # CHANGED: The critique must be aware of the original context.
        prompt = f"""{context}

        --- RESPONSE TO CRITIQUE ---
        {answer}
        
        Critique the answer above based on the user's query and the full context provided. Provide scores from 0.0 to 1.0 for the following criteria:
        - Truth: Is it accurate and factual?
        - Helpfulness: Does it fully address the user's intent?
        - Clarity: Is it well-structured and easy to understand?
        
        CRITIQUE SCORES:"""
        
        critique_response = self.model_manager.generate_text(pathway.critique_model, prompt, max_tokens=100)
        
        scores = {}
        for line in critique_response.split('\n'):
            match = re.search(r'(\w+):\s*([0-9.]+)', line, re.IGNORECASE)
            if match:
                key, value = match.groups()
                try:
                    score_val = float(value)
                    scores[key.lower()] = max(0.0, min(1.0, score_val)) # Clamp score between 0 and 1
                except ValueError:
                    continue
        
        for key in ['truth', 'helpfulness', 'clarity']:
            if key not in scores: scores[key] = 0.5
        
        return scores

    # CHANGED: Added 'context' parameter to the main reason method.
    def reason(self, query: str, context: str, progress_callback: Optional[Callable[[str, Any], None]] = None, gear_override: Optional[str] = None) -> Trace:
        """The main entry point for the reasoning process."""
        logger.info(f"ATPv3 starting reasoning for query: '{query[:80]}...'")
        
        # 1. Conductor: Triage and Determine Pathway
        if progress_callback: progress_callback("triage", "Conductor is assessing the query...")
        triage_result = self.conductor.conduct_triage(query, gear_override)
        pathway = self.conductor.determine_cognitive_pathway(query)
        # Pass context along with pathway info for better telemetry
        if progress_callback: progress_callback("triage", {"triage": triage_result, "pathway": pathway.model_dump(), "context": context})

        # --- Gear 1: Direct Response Pathway ---
        if triage_result["recommended_gear"] == 'gear_1':
            return self._generate_gear_1_response(query, context, pathway, progress_callback)

        # --- Gear 2/3: Cognitive Weaving Pathway ---
        start_time = datetime.now()
        
        if progress_callback: progress_callback("plan", f"Sketching plan with {pathway.execution_model}...")
        initial_plan = self._generate_plan(context, pathway) # Pass context
        if progress_callback: progress_callback("plan", {"initial_plan": initial_plan})
        
        if progress_callback: progress_callback("enrich", f"Enriching plan with {pathway.plan_enrichment_model}...")
        final_plan = self._enrich_plan(initial_plan, context, pathway) # Pass context
        if progress_callback: progress_callback("enrich", {"final_plan": final_plan})
        
        if progress_callback: progress_callback("execute", f"Executing with {pathway.execution_model}...")
        answer = self._execute_plan(context, final_plan, pathway) # Pass context
        if progress_callback: progress_callback("execute", {"answer_preview": answer[:100], "pathway": pathway.model_dump()})
        
        if progress_callback: progress_callback("critique", f"Critiquing with {pathway.critique_model}...")
        scores = self._critique_response(context, answer, pathway) # Pass context
        if progress_callback: progress_callback("critique", {"scores": scores})

        # 6. Assemble Final Trace
        trace_id = f"trace_{uuid.uuid4()}"
        timestamp = datetime.now(timezone.utc).isoformat()
        total_score = sum(self.identity.weights.get(k, 0.0) * v for k, v in scores.items())
        
        attempt = Attempt(id=1, plan=final_plan, candidate=answer, scores=scores, total=total_score)
        
        reasoning_str = f"Generated via Cognitive Weaving. Primary Reasoner: {pathway.execution_model}. Plan Supervisor: {pathway.plan_enrichment_model}."
        summary = Summary(answer=answer, reasoning=reasoning_str, next_action="Awaiting next user query.")
        
        model_info = ModelInfo(
            name=f"Orchestra: {pathway.execution_model} + {pathway.plan_enrichment_model}",
            runtime="local" if not self.model_manager.dummy_mode else "dummy",
            temp=0.2
        )
        
        reflection = Reflection(
            what_worked="The multi-model 'Cognitive Weaving' process provided a structured and detailed response.",
            what_failed="Potential for latency due to multiple model calls. Resource management for swapping models needs to be optimized.",
            next_adjustment="Implement intelligent model loading/unloading in the ModelManager."
        )

        trace = Trace(
            trace_id=trace_id,
            query=query,
            seed_id=self.identity.seed_id,
            summary=summary,
            attempts=[attempt],
            best=Best(attempt_id=1, candidate=answer, total=total_score),
            reflection=reflection,
            model_info=model_info,
            timestamp=timestamp
        )
        
        self.memory.save_trace(trace)
        total_time = (datetime.now() - start_time).total_seconds()
        logger.success(f"Cognitive Weaving trace {trace.trace_id} saved in {total_time:.2f}s.")
        return trace