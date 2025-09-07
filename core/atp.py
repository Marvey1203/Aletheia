# core/atp.py
# Enhanced Scaffolded Generation Protocol with Adaptive Reasoning Depth
# ATP v2.1 - Cognitive Gearbox Implementation with Windows-compatible timeout

import uuid
import json
import re
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from typing import List, Dict, Tuple, Optional, Callable, Any, Literal
from enum import Enum

from loguru import logger
from pydantic import ValidationError

from .schemas import Trace, Summary, Attempt, Best, Reflection, ModelInfo
from .identity import IdentityCore
from .memory import MemoryGalaxy
from .llm import LocalLLM

class ReasoningStage(Enum):
    TRIAGE = "triage"
    PLAN = "plan"
    EXECUTE = "execute"
    CRITIQUE = "critique"
    REFLECTION = "reflection"

class ReasoningGear(Enum):
    GEAR_1 = "gear_1"  # Simple responses (greetings, trivial questions)
    GEAR_2 = "gear_2"  # Standard scaffolded protocol
    GEAR_3 = "gear_3"  # Deep analysis (Polymath Protocol - future implementation)

class ATPLoopV2:
    """
    Enhanced Scaffolded Generation Protocol with Adaptive Reasoning Depth.
    Implements a Cognitive Gearbox that adjusts reasoning depth based on query complexity.
    """
    
    # Output templates for each stage (optimized for small models)
    STAGE_TEMPLATES = {
        ReasoningStage.TRIAGE: (
            "TRIAGE:\n"
            "Complexity: {complexity_level}\n"
            "Recommended Gear: {recommended_gear}\n"
            "Reason: {reason}\n\n"
            "END_TRIAGE"
        ),
        ReasoningStage.PLAN: (
            "PLAN:\n"
            "1. {step1}\n"
            "2. {step2}\n"
            "3. {step3}\n"
            "4. {step4}\n"
            "5. {step5}\n\n"
            "END_PLAN"
        ),
        ReasoningStage.EXECUTE: (
            "EXECUTION:\n"
            "Following the plan: {plan}\n\n"
            "ANSWER: {answer}\n\n"
            "END_EXECUTION"
        ),
        ReasoningStage.CRITIQUE: (
            "CRITIQUE:\n"
            "Truth: {truth_score}\n"
            "Helpfulness: {helpfulness_score}\n"
            "Clarity: {clarity_score}\n"
            "Ethics: {ethics_score}\n"
            "Curiosity: {curiosity_score}\n\n"
            "END_CRITIQUE"
        )
    }

    def __init__(self, identity: IdentityCore, memory: MemoryGalaxy, llm: LocalLLM):
        self.identity = identity
        self.memory = memory
        self.llm = llm
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.greeting_cache = {}
        logger.info("ATP Loop v2.1 (Adaptive Reasoning Depth) initialized")

    def _parse_stage_output(self, stage: ReasoningStage, output: str) -> dict:
        """Enhanced parsing using IRP techniques for each stage"""
        try:
            if stage == ReasoningStage.TRIAGE:
                complexity_match = re.search(r"Complexity:\s*(.+)", output, re.IGNORECASE)
                gear_match = re.search(r"Recommended Gear:\s*(.+)", output, re.IGNORECASE)
                reason_match = re.search(r"Reason:\s*(.+)", output, re.IGNORECASE)
                
                if complexity_match and gear_match:
                    return {
                        "complexity": complexity_match.group(1).strip(),
                        "recommended_gear": gear_match.group(1).strip(),
                        "reason": reason_match.group(1).strip() if reason_match else ""
                    }
                
            elif stage == ReasoningStage.PLAN:
                # Try multiple patterns to extract plan
                patterns = [
                    r"PLAN:\n((?:\d\..+?\n)+)",
                    r"Plan:\n((?:\d\..+?\n)+)",
                    r"Steps:\n((?:\d\..+?\n)+)",
                    r"((?:\d\..+?\n)+)"  # Just look for any numbered list
                ]
                
                for pattern in patterns:
                    plan_match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                    if plan_match:
                        steps = [step.strip() for step in plan_match.group(1).split('\n') 
                                if step.strip() and re.match(r"^\d+\.", step)]
                        if steps:
                            return {"plan": steps}
                
                # Fallback: split by newline and look for any structured content
                lines = [line.strip() for line in output.split('\n') if line.strip()]
                if lines:
                    return {"plan": [f"{i+1}. {line}" for i, line in enumerate(lines[:3])]}
                    
            elif stage == ReasoningStage.EXECUTE:
                # Try multiple patterns to extract answer
                patterns = [
                    r"ANSWER:\s*(.+?)(?=\n\n|\n*$)",
                    r"Answer:\s*(.+?)(?=\n\n|\n*$)",
                    r"Response:\s*(.+?)(?=\n\n|\n*$)",
                    r"^(.+?)(?=\n\n|\n*$)"  # Just take the first paragraph
                ]
                
                for pattern in patterns:
                    answer_match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                    if answer_match:
                        return {"answer": answer_match.group(1).strip()}
                
                # Fallback: return the entire output
                return {"answer": output.strip()}
                    
            elif stage == ReasoningStage.CRITIQUE:
                scores = {}
                # Look for score patterns in various formats
                for line in output.split('\n'):
                    if any(metric in line.lower() for metric in 
                         ['truth', 'helpfulness', 'clarity', 'ethics', 'curiosity']):
                        # Try different delimiters
                        for delimiter in [':', '=', '-']:
                            if delimiter in line:
                                parts = line.split(delimiter, 1)
                                if len(parts) == 2:
                                    key = parts[0].strip().lower()
                                    value = parts[1].strip()
                                    try:
                                        # Extract first number from the value
                                        num_match = re.search(r"[-+]?\d*\.\d+|\d+", value)
                                        if num_match:
                                            scores[key] = float(num_match.group())
                                            break
                                    except ValueError:
                                        continue
                
                # Ensure we have all required scores with defaults
                required_scores = ['truth', 'helpfulness', 'clarity', 'ethics', 'curiosity']
                for score in required_scores:
                    if score not in scores:
                        scores[score] = 0.5  # Default value
                        
                return {"scores": scores}
                    
        except Exception as e:
            logger.warning(f"Parsing failed for {stage}: {e}")
        
        # Return appropriate defaults for each stage
        if stage == ReasoningStage.PLAN:
            return {"plan": ["1. Analyze the query", "2. Formulate response", "3. Verify accuracy"]}
        elif stage == ReasoningStage.EXECUTE:
            return {"answer": "I need more time to think about this. Please try again."}
        elif stage == ReasoningStage.CRITIQUE:
            return {"scores": {"truth": 0.5, "helpfulness": 0.5, "clarity": 0.5, "ethics": 0.5, "curiosity": 0.0}}
        
        return {}

    def _triage_query(self, query: str) -> Dict[str, str]:
        """Triage stage to determine query complexity and appropriate reasoning gear"""
        # Check if it's a simple greeting first (avoid LLM call for common greetings)
        query_lower = query.lower().strip()
        simple_greetings = ["hi", "hello", "hey", "howdy", "greetings", "what's up", 
                           "good morning", "good afternoon", "good evening", 
                           "thanks", "thank you", "bye", "goodbye"]
        
        if any(query_lower.startswith(greeting) for greeting in simple_greetings):
            return {
                "complexity": "simple",
                "recommended_gear": "gear_1",
                "reason": "Recognized as simple greeting"
            }
        
        triage_prompt = f"""Analyze this query and determine its complexity level:

Query: "{query}"

Classify it into one of these complexity levels:
- "simple": Greetings, pleasantries, or trivial questions that can be answered directly
- "moderate": Factual questions or requests that require some reasoning
- "complex": Deep philosophical questions, ethical dilemmas, or complex problem-solving

Use this format:
{self.STAGE_TEMPLATES[ReasoningStage.TRIAGE]}

Be concise but accurate in your assessment."""
        
        try:
            response = self.llm.generate_text(triage_prompt, max_tokens=100)
            parsed = self._parse_stage_output(ReasoningStage.TRIAGE, response)
            
            # Default to moderate complexity if parsing fails
            if not parsed:
                return {
                    "complexity": "moderate",
                    "recommended_gear": "gear_2",
                    "reason": "Default fallback for triage failure"
                }
            
            # Map complexity to gear
            complexity_to_gear = {
                "simple": "gear_1",
                "moderate": "gear_2",
                "complex": "gear_3"
            }
            
            # Ensure we have a valid gear recommendation
            complexity = parsed.get("complexity", "").lower()
            recommended_gear = complexity_to_gear.get(complexity, "gear_2")
            
            return {
                "complexity": parsed.get("complexity", "moderate"),
                "recommended_gear": recommended_gear,
                "reason": parsed.get("reason", "Complexity assessment completed")
            }
        except Exception as e:
            logger.warning(f"Triage failed: {e}")
            return {
                "complexity": "moderate",
                "recommended_gear": "gear_2",
                "reason": f"Triage error: {str(e)}"
            }

    def _generate_gear_1_response(self, query: str) -> str:
        """Generate a direct response for simple queries (Gear 1)"""
        # Check cache first
        query_lower = query.lower().strip()
        if query_lower in self.greeting_cache:
            return self.greeting_cache[query_lower]
        
        # Common greetings and responses
        greeting_responses = {
            "hi": "Hello! How can I assist you today?",
            "hello": "Hello! I'm Aletheia, your sovereign AI partner. How can I help?",
            "hey": "Hey there! What can I do for you?",
            "howdy": "Howdy! Ready to explore some ideas together?",
            "greetings": "Greetings! I'm here to help with your questions and thoughts.",
            "what's up": "Not much, just ready to help you! What's on your mind?",
            "good morning": "Good morning! A great day for thinking and exploration.",
            "good afternoon": "Good afternoon! How can I assist you today?",
            "good evening": "Good evening! What would you like to discuss?",
            "thanks": "You're welcome! Is there anything else I can help with?",
            "thank you": "You're welcome! Happy to assist.",
            "bye": "Goodbye! Feel free to return whenever you have more questions.",
            "goodbye": "Goodbye! It was a pleasure assisting you."
        }
        
        # Check for exact matches first
        if query_lower in greeting_responses:
            response = greeting_responses[query_lower]
            self.greeting_cache[query_lower] = response
            return response
        
        # Check for queries that start with greetings
        for greeting in greeting_responses:
            if query_lower.startswith(greeting):
                response = greeting_responses[greeting]
                self.greeting_cache[query_lower] = response
                return response
        
        # For other simple queries, use a direct approach
        try:
            direct_prompt = f"""Provide a concise, direct answer to this simple query:

Query: "{query}"

Answer directly without a plan or complex reasoning."""
            
            response = self.llm.generate_text(direct_prompt, max_tokens=100)
            self.greeting_cache[query_lower] = response
            return response
        except Exception as e:
            logger.warning(f"Gear 1 response generation failed: {e}")
            return "I'm here to help! What would you like to know?"

    def _generate_plan(self, query: str) -> List[str]:
        """Plan stage with optimized prompt and parsing"""
        plan_prompt = f"""Create a step-by-step plan to answer: '{query}'

Use this format:
{self.STAGE_TEMPLATES[ReasoningStage.PLAN]}

Be concise but thorough."""
        
        try:
            response = self.llm.generate_text(plan_prompt, max_tokens=150)
            parsed = self._parse_stage_output(ReasoningStage.PLAN, response)
            return parsed.get("plan", ["1. Analyze the query", "2. Formulate response", "3. Verify accuracy"])
        except Exception as e:
            logger.warning(f"Plan generation failed: {e}")
            return ["1. Analyze the query", "2. Formulate response", "3. Verify accuracy"]

    def _execute_plan(self, query: str, plan: List[str]) -> str:
        """Execute stage with structured output"""
        exec_prompt = f"""Execute this plan to answer: '{query}'

Plan: {plan}

Use this format:
{self.STAGE_TEMPLATES[ReasoningStage.EXECUTE].format(plan=plan, answer='{answer}')}

Provide a clear, direct answer."""
        
        try:
            response = self.llm.generate_text(exec_prompt, max_tokens=200)
            parsed = self._parse_stage_output(ReasoningStage.EXECUTE, response)
            return parsed.get("answer", "I need more time to think about this. Please try again.")
        except Exception as e:
            logger.warning(f"Execution failed: {e}")
            return "I'm experiencing difficulties processing your request. Please try again."

    def _critique_response(self, query: str, plan: List[str], answer: str) -> Dict[str, float]:
        """Critique stage with parallel scoring and score normalization."""
        critique_prompt = f"""Critique this answer for query: '{query}'

Plan used: {plan}
Answer: {answer}

Use this format:
{self.STAGE_TEMPLATES[ReasoningStage.CRITIQUE]}

Score each category as a float between 0.0 (terrible) and 1.0 (perfect)."""
        
        try:
            response = self.llm.generate_text(critique_prompt, max_tokens=100)
            parsed = self._parse_stage_output(ReasoningStage.CRITIQUE, response)
            scores = parsed.get("scores", {})

            # --- BEGIN FIX: SCORE NORMALIZATION ---
            normalized_scores = {}
            for key, value in scores.items():
                try:
                    score_float = float(value)
                    # If the model gives a score from 1-10, divide by 10.
                    if score_float > 1.0:
                        normalized_scores[key] = score_float / 10.0
                    # Ensure the score is not negative.
                    elif score_float < 0.0:
                        normalized_scores[key] = 0.0
                    else:
                        normalized_scores[key] = score_float
                except (ValueError, TypeError):
                    # If the score is not a valid number, default to 0.5.
                    normalized_scores[key] = 0.5
            
            # Ensure all required keys are present
            for required_key in ['truth', 'helpfulness', 'clarity', 'ethics', 'curiosity']:
                if required_key not in normalized_scores:
                    normalized_scores[required_key] = 0.5

            return normalized_scores
            # --- END FIX ---

        except Exception as e:
            logger.warning(f"Critique failed: {e}")
            return {"truth": 0.5, "helpfulness": 0.5, "clarity": 0.5, "ethics": 0.5, "curiosity": 0.0}
        
    def _generate_reflection_parallel(self, plan: List[str], answer: str, scores: Dict[str, float]) -> Reflection:
        """Generate reflection in parallel using executor"""
        def create_reflection():
            # Simple rule-based reflection that can be enhanced later
            high_scores = all(score > 0.7 for score in scores.values() if score != 'curiosity')
            low_scores = any(score < 0.4 for score in scores.values() if score != 'curiosity')
            
            return Reflection(
                what_worked="Structured reasoning process provided reliable output",
                what_failed="Potential areas for improvement in scoring accuracy" if low_scores else "No significant failures",
                next_adjustment="Enhance scoring criteria and reflection depth" if low_scores else "Maintain current approach"
            )
        
        return self.executor.submit(create_reflection).result()

    def _reason_gear_1(self, query: str, progress_callback: Optional[Callable[[str, Any], None]] = None) -> Trace:
        """Gear 1 reasoning: Direct response for simple queries"""
        if progress_callback:
            progress_callback("gear_1", "Generating direct response...")
        
        answer = self._generate_gear_1_response(query)
        
        # Create a simplified trace for gear 1
        trace_id = f"trace_{uuid.uuid4()}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        attempt = Attempt(
            id=1,
            plan=["1. Direct response generation"],
            candidate=answer,
            scores={"truth": 0.9, "helpfulness": 0.9, "clarity": 0.9, "ethics": 0.9, "curiosity": 0.0},
            total=0.9
        )
        
        summary = Summary(
            answer=answer,
            reasoning="Generated using Gear 1 (Direct Response) for simple query",
            next_action="No further action needed for simple query"
        )
        
        model_info = ModelInfo(
            name=self.llm.model_path.name if self.llm.model_path else "dummy",
            quant="Q4_K_M",
            runtime="local" if not self.llm.dummy_mode else "dummy_mode",
            temp=0.2
        )

        reflection = Reflection(
            what_worked="Direct response efficiently handled simple query",
            what_failed="N/A for Gear 1 reasoning",
            next_adjustment="Maintain current approach for simple queries"
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
        return trace

    def _reason_gear_2(self, query: str, progress_callback: Optional[Callable[[str, Any], None]] = None) -> Trace:
        """Gear 2 reasoning: Standard scaffolded protocol"""
        # Maintain the cognitive process: Plan → Execute → Critique
        start_time = datetime.now()
        
        # 1. Plan Stage (Cognitive Foundation)
        if progress_callback:
            progress_callback("plan", "Starting planning phase...")
        plan = self._generate_plan(query)
        if progress_callback:
            progress_callback("plan", plan)
        logger.info(f"Generated Plan: {plan}")
        
        # 2. Execute Stage
        if progress_callback:
            progress_callback("execute", "Starting execution phase...")
        answer = self._execute_plan(query, plan)
        if progress_callback:
            progress_callback("execute", answer)
        
        # 3. Critique Stage 
        if progress_callback:
            progress_callback("critique", "Starting critique phase...")
        scores = self._critique_response(query, plan, answer)
        if progress_callback:
            progress_callback("critique", scores)
        
        # 4. Calculate total score and generate reflection in parallel
        total_score = sum(self.identity.weights.get(k, 0) * v 
                        for k, v in scores.items() if k != 'curiosity')
        total_score += scores.get('curiosity', 0)
        
        if progress_callback:
            progress_callback("reflection", "Generating reflection...")
        reflection = self._generate_reflection_parallel(plan, answer, scores)
        if progress_callback:
            progress_callback("reflection", reflection)
        
        # 5. Assemble final trace (Maintaining full auditability)
        trace_id = f"trace_{uuid.uuid4()}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        attempt = Attempt(
            id=1,
            plan=plan,
            candidate=answer,
            scores=scores,
            total=total_score
        )
        
        summary = Summary(
            answer=answer,
            reasoning="Generated using Gear 2 (Scaffolded Protocol) with parallel optimization",
            next_action="Review detailed trace in Observatory"
        )
        
        model_info = ModelInfo(
            name=self.llm.model_path.name if self.llm.model_path else "dummy",
            quant="Q4_K_M",
            runtime="local" if not self.llm.dummy_mode else "dummy_mode",
            temp=0.2
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
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.success(f"Gear 2 Reasoning completed in {total_time:.2f}s")
        
        self.memory.save_trace(trace)
        return trace

    def _reason_gear_3(self, query: str, progress_callback: Optional[Callable[[str, Any], None]] = None) -> Trace:
        """Gear 3 reasoning: Deep analysis using the Manifold V2 Polymath Protocol."""
        logger.info("Engaging Gear 3: Polymath Protocol")
        
        if progress_callback:
            progress_callback("gear_3", "Initializing Council of Experts...")

        # For V1 of the Polymath, the personas are fixed. V1.5 will make them dynamic.
        persona_ids = ["Architect", "Critic", "Strategist"]
        
        # --- 1. The Parallel Debate ---
        # We run a full, independent Gear 2 reasoning cycle for each persona in parallel.
        sub_traces: List[Trace] = []
        
        def run_persona_reasoning(persona_id: str) -> Trace:
            # Create a temporary, specialized IdentityCore for this persona
            # In the future, this would load a custom `persona.json` file
            persona_identity = IdentityCore(seed_id=self.identity.seed_id)
            persona_identity.principles.insert(0, f"You are The {persona_id}. Your reasoning must be guided by this role.")
            
            # Create a temporary ATP loop for this persona
            persona_atp = ATPLoopV2(persona_identity, self.memory, self.llm)
            
            # We add the persona_id to the query to give the model context
            persona_query = f"As The {persona_id}, analyze this query: '{query}'"
            
            # Each persona uses the efficient Gear 2 protocol
            trace = persona_atp._reason_gear_2(persona_query)
            trace.persona_id = persona_id # Tag the trace with its persona
            return trace

        with ThreadPoolExecutor(max_workers=len(persona_ids)) as executor:
            if progress_callback:
                progress_callback("gear_3", f"Debate started with {persona_ids}...")
            
            # Map each persona to the reasoning function
            future_to_persona = {executor.submit(run_persona_reasoning, pid): pid for pid in persona_ids}

            # We use `as_completed` to process results as they finish.
            for future in as_completed(future_to_persona):
                persona_id = future_to_persona[future]
                try:
                    result = future.result(timeout=180) # 3-minute timeout per persona
                    sub_traces.append(result)
                except FutureTimeoutError:
                    logger.error(f"Persona '{persona_id}' timed out during reasoning.")
                    error_trace = self._create_fallback_trace(query, f"Persona '{persona_id}' timed out.")
                    error_trace.persona_id = persona_id
                    sub_traces.append(error_trace)
                except Exception as e:
                    logger.error(f"Persona '{persona_id}' failed with an exception: {e}")
                    error_trace = self._create_fallback_trace(query, f"Persona '{persona_id}' failed: {e}")
                    error_trace.persona_id = persona_id
                    sub_traces.append(error_trace)

        if progress_callback:
            progress_callback("gear_3", "Debate complete. Synthesizing results...")

        # --- 2. The Arbiter's Synthesis ---
        # The Arbiter now receives the winning candidates from the council's debate.
        
        arbiter_context = "\n".join([
            f"--- Proposal from The {trace.persona_id} ---\n{trace.best.candidate}\n"
            for trace in sub_traces
        ])

        arbiter_query = f"""
        You are The Arbiter. You have received the following proposals from your council of experts in response to the original query: '{query}'

        {arbiter_context}

        Your task is to synthesize these competing perspectives into a single, superior, and actionable final answer. Do not just list the options; find the deeper insight that integrates the best parts of each.
        """
        
        # The Arbiter also uses the fast Gear 2 protocol to form its final opinion.
        arbiter_trace = self._reason_gear_2(arbiter_query)
        
        # --- 3. Final Assembly ---
        # We attach the council's sub_traces to the final Arbiter trace for full auditability.
        arbiter_trace.sub_traces = sub_traces
        arbiter_trace.persona_id = "Arbiter"
        arbiter_trace.summary.reasoning = "Generated using Gear 3 (Polymath Protocol) with a council of experts."

        logger.success("Polymath Protocol completed successfully.")
        
        # We save the final, nested trace to memory
        self.memory.save_trace(arbiter_trace)
        
        return arbiter_trace

    def reason(self, query: str, progress_callback: Optional[Callable[[str, Any], None]] = None, 
               gear_override: Optional[str] = None) -> Trace:
        """
        Enhanced reasoning cycle with Adaptive Reasoning Depth.
        Uses a Cognitive Gearbox to adjust reasoning depth based on query complexity.
        """
        logger.info(f"Initiating adaptive reasoning for: '{query}'")
        
        try:
            # 0. Triage Stage - Determine query complexity and appropriate gear
            if progress_callback:
                progress_callback("triage", "Analyzing query complexity...")
            
            # Use override if provided, otherwise perform triage
            if gear_override and gear_override in ["gear_1", "gear_2", "gear_3"]:
                triage_result = {
                    "complexity": "override",
                    "recommended_gear": gear_override,
                    "reason": f"User manually overrode to {gear_override}"
                }
            else:
                triage_result = self._triage_query(query)
            
            recommended_gear = triage_result["recommended_gear"]
            if progress_callback:
                progress_callback("triage", f"Recommended gear: {recommended_gear} - {triage_result['reason']}")
            
            logger.info(f"Query complexity: {triage_result['complexity']}, Recommended gear: {recommended_gear}")
            
            # Route to appropriate gear
            if recommended_gear == "gear_1":
                return self._reason_gear_1(query, progress_callback)
            elif recommended_gear == "gear_2":
                return self._reason_gear_2(query, progress_callback)
            elif recommended_gear == "gear_3":
                return self._reason_gear_3(query, progress_callback)
            else:
                # Fallback to Gear 2
                logger.warning(f"Unknown gear recommended: {recommended_gear}, defaulting to Gear 2")
                return self._reason_gear_2(query, progress_callback)

        except Exception as e:
            logger.error(f"Adaptive reasoning failed: {e}")
            # Create a simple fallback trace instead of trying to import the old ATPLoop
            return self._create_fallback_trace(query, str(e))

    def _create_fallback_trace(self, query: str, error: str) -> Trace:
        """Create a fallback trace when reasoning fails"""
        trace_id = f"trace_{uuid.uuid4()}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        attempt = Attempt(
            id=1,
            plan=["1. Error occurred during reasoning"],
            candidate=f"Error: {error}",
            scores={"truth": 0.1, "helpfulness": 0.1, "clarity": 0.1, "ethics": 0.1, "curiosity": 0.0},
            total=0.1
        )
        
        summary = Summary(
            answer="An error occurred during reasoning. Please check the logs.",
            reasoning="The adaptive reasoning protocol encountered an error and could not complete.",
            next_action="Check system logs and try again with a simpler query."
        )
        
        model_info = ModelInfo(
            name=self.llm.model_path.name if self.llm.model_path else "dummy",
            quant="Q4_K_M",
            runtime="local" if not self.llm.dummy_mode else "dummy_mode",
            temp=0.2
        )

        reflection = Reflection(
            what_worked="Fallback mechanism provided an error response",
            what_failed=f"Reasoning process failed with error: {error}",
            next_adjustment="Investigate and fix the underlying issue in the reasoning protocol"
        )

        return Trace(
            trace_id=trace_id,
            query=query,
            seed_id=self.identity.seed_id,
            summary=summary,
            attempts=[attempt],
            best=Best(attempt_id=1, candidate=attempt.candidate, total=attempt.total),
            reflection=reflection,
            model_info=model_info,
            timestamp=timestamp
        )