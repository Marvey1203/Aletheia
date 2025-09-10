import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger
from langgraph.graph import StateGraph, END
from sentence_transformers import util

from .schemas import GraphState, PlanStep
from .identity import IdentityCore
from .llm import ModelManager
from .orchestrator import Conductor

# --- Configuration ---
CONSTITUTIONAL_VECTOR_PATH = Path("core_knowledge/constitutional_vector.npy")
SOCIAL_VECTORS_PATH = Path("core_knowledge/social_vectors.npz")
REVISION_THRESHOLD = 0.6 

# This library is now for decoding the ACTION_TYPE of a PlanStep
PLAN_ACTION_LIBRARY = {
    "ACKNOWLEDGE": "Acknowledge the user's statement and social context.",
    "GREET": "Greet the user in a friendly and appropriate manner.",
    "ANALYZE": "Analyze the core subject of the user's query.",
    "RECALL": "Recall relevant information from the Memory Galaxy.",
    "SYNTHESIZE": "Synthesize my core principles and recalled memories to form a conclusion.",
    "FORMULATE": "Formulate the final, comprehensive answer based on the synthesized concepts.",
    "INQUIRE": "Ask an insightful, clarifying question to expand our mutual understanding."
}

# --- The Cognitive Graph Nodes ---

def social_acuity_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """Creates a rich 'social map' of the user's query."""
    logger.info("OmegaNode: Social Acuity")
    if not SOCIAL_VECTORS_PATH.exists():
        logger.error("Social Vectors not found.")
        return {"social_context": {}}
    social_vectors = np.load(SOCIAL_VECTORS_PATH)
    query = state["query"]
    query_vector = model_manager.create_embedding(query)
    social_context = {}
    for name, vector in social_vectors.items():
        similarity = util.cos_sim(query_vector, vector).item()
        social_context[name] = similarity
    logger.info(f"Social Context: { {k: f'{v:.2f}' for k, v in social_context.items()} }")
    return {"social_context": social_context}

def determine_pathway_node(state: GraphState, conductor: Conductor) -> Dict[str, Any]:
    """Determines the cognitive pathway (which models to use) for the query."""
    logger.info("Node: determine_pathway")
    query = state["query"]
    pathway = conductor.determine_cognitive_pathway(query)
    return {"pathway": pathway.model_dump()}

# --- NEW OMEGA PLANNER V2 ---
def omega_planner_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """
    Constructs a structured, conceptual plan using PlanStep objects.
    This is the implementation of Semantic Scaffolding.
    """
    logger.info("OmegaNode: Planner V2 (Structured)")
    
    query = state["query"]
    social_context = state["social_context"]
    
    # 1. Encode core concepts for this thought process
    # --- THIS IS THE CRITICAL FIX ---
    # Convert the output of the embedding model (a list) into a NumPy array
    # at the moment of creation to ensure type safety.
    query_vec = np.array(model_manager.create_embedding(query), dtype=np.float32)
    plan_action_vectors = {
        name: np.array(model_manager.create_embedding(desc), dtype=np.float32) 
        for name, desc in PLAN_ACTION_LIBRARY.items()
    }
    # --- END OF FIX ---

    # 2. The Planning Logic (a more advanced, rule-based Orchestrator)
    conceptual_plan = []
    
    # Rule 1: Always start by acknowledging the user.
    conceptual_plan.append(PlanStep(
        action_type=plan_action_vectors["ACKNOWLEDGE"],
        action_subject=query_vec
    ))

    # Rule 2: If the intent is Meta-Cognition, the plan is to analyze and synthesize.
    if social_context.get("META_COGNITION", 0.0) > 0.3:
        logger.info("Planner: Meta-cognition path selected.")
        conceptual_plan.append(PlanStep(
            action_type=plan_action_vectors["ANALYZE"],
            action_subject=np.array(model_manager.create_embedding("my own planning process"), dtype=np.float32)
        ))
        conceptual_plan.append(PlanStep(
            action_type=plan_action_vectors["SYNTHESIZE"]
        ))
        conceptual_plan.append(PlanStep(
            action_type=plan_action_vectors["FORMULATE"]
        ))
    # Rule 3: If casual, the plan is simple.
    elif social_context.get("CASUALNESS", 0.0) > 0.4:
        logger.info("Planner: Casual path selected.")
        conceptual_plan.append(PlanStep(
            action_type=plan_action_vectors["GREET"]
        ))
        conceptual_plan.append(PlanStep(
            action_type=plan_action_vectors["INQUIRE"]
        ))
    # Default Rule: Standard query processing.
    else:
        logger.info("Planner: Standard query path selected.")
        conceptual_plan.append(PlanStep(
            action_type=plan_action_vectors["ANALYZE"],
            action_subject=query_vec
        ))
        conceptual_plan.append(PlanStep(
            action_type=plan_action_vectors["RECALL"]
        ))
        conceptual_plan.append(PlanStep(
            action_type=plan_action_vectors["FORMULATE"]
        ))

    logger.success(f"Omega Planner constructed a conceptual plan with {len(conceptual_plan)} structured steps.")
    return {"conceptual_plan": conceptual_plan}

# --- NEW PLAN DECODER V2 ---
def plan_decoder_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """
    Translates the structured conceptual plan into a high-fidelity linguistic plan.
    """
    logger.info("Node: Plan Decoder V2 (High-Fidelity)")

    conceptual_plan = state["conceptual_plan"]
    
    action_names = list(PLAN_ACTION_LIBRARY.keys())
    action_vectors = [model_manager.create_embedding(desc) for desc in PLAN_ACTION_LIBRARY.values()]

    linguistic_plan = []
    for i, step in enumerate(conceptual_plan):
        # Decode the action_type
        sims = util.cos_sim(step.action_type, action_vectors)[0]
        decoded_action = action_names[sims.argmax().item()]

        # Decode the action_subject (if it exists)
        # This is a simplification; a real decoder would use an LLM for this.
        # For now, we just represent it conceptually.
        subject_text = ""
        if step.action_subject is not None:
            # A more advanced version would compare the subject vector to known entities.
            # Here, we just acknowledge its presence.
            subject_text = " on the user's query"

        linguistic_step = f"Step {i+1}: {decoded_action.replace('_', ' ').capitalize()}{subject_text}."
        linguistic_plan.append(linguistic_step)
        logger.info(f"  - Decoded Step {i}: {linguistic_step}")

    logger.success(f"Decoded conceptual plan into {len(linguistic_plan)} linguistic steps.")
    return {"linguistic_plan": linguistic_plan}


def execute_model_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """Executes the high-fidelity linguistic plan to generate a candidate answer."""
    logger.info("Node: execute_model")
    context = state["context"]
    pathway = state["pathway"]
    # UPDATED: Use the new linguistic_plan key
    plan_str = "\n".join(state["linguistic_plan"])

    prompt = f"""[INST]
You are Aletheia. The following is your internal monologue and plan for how to respond to the user. This is for your eyes only. Do not repeat or narrate this plan in your final answer.

--- Internal Plan ---
{plan_str}
--- End Internal Plan ---

Based on this plan, generate your final, user-facing response. Address the user directly.
[/INST]

FINAL RESPONSE:"""
    
    answer = model_manager.generate_text(pathway['execution_model'], prompt, max_tokens=1500)
    return {"candidate_answer": answer}

# --- The rest of the nodes (critique, should_finish, revise) are unchanged for this sprint ---
# --- They will be upgraded in the future to use the new structured plan ---

def omega_critique_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    logger.info("OmegaNode: critique")
    if not CONSTITUTIONAL_VECTOR_PATH.exists():
        return {"scores": {"constitutional_alignment": 0.5}}
    constitutional_vector = np.load(CONSTITUTIONAL_VECTOR_PATH)
    answer = state["candidate_answer"]
    answer_vector = model_manager.create_embedding(answer)
    similarity_score = util.cos_sim(answer_vector, constitutional_vector).item()
    logger.success(f"Omega Critique complete. Constitutional Alignment: {similarity_score:.4f}")
    return {"scores": {"constitutional_alignment": similarity_score}}

def should_finish_node(state: GraphState) -> str:
    logger.info("Edge: should_finish_principled")
    scores = state.get("scores", {})
    alignment_score = scores.get("constitutional_alignment", 0.0)
    revision_history = state.get("revision_history", [])
    if alignment_score >= REVISION_THRESHOLD:
        return "end"
    if len(revision_history) >= 2:
        return "end"
    return "revise"

def revise_plan_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    logger.info("Node: revise_plan (Integrated)")
    # This node will need to be upgraded in a future sprint to re-run the V2 planner
    # For now, it will fall back to the LLM-based revision.
    context = state["context"]
    pathway = state["pathway"]
    last_answer = state["candidate_answer"]
    alignment_score = state["scores"]["constitutional_alignment"]
    social_context = state.get("social_context", {})
    dominant_social_context = max(social_context, key=social_context.get) if social_context else "standard"
    revision_history = state.get("revision_history", [])
    revision_history.append(f"Attempt failed with Alignment: {alignment_score:.4f}. Social Context: {dominant_social_context}.")
    prompt = f"[INST]...NEW, INTEGRATED PLAN:[/INST]" # Simplified for brevity
    response = model_manager.generate_text(pathway['plan_enrichment_model'], prompt, max_tokens=300)
    new_plan = [line.strip() for line in response.split('\n') if line.strip()]
    return {"linguistic_plan": new_plan, "revision_history": revision_history}

# --- The Graph Builder ---
def build_cognitive_graph(identity: IdentityCore, model_manager: ModelManager, conductor: Conductor):
    """
    Builds the LangGraph-based cognitive process, now featuring the Omega Planner V2.
    """
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("social_acuity", lambda state: social_acuity_node(state, model_manager))
    workflow.add_node("determine_pathway", lambda state: determine_pathway_node(state, conductor))
    workflow.add_node("omega_planner", lambda state: omega_planner_node(state, model_manager))
    workflow.add_node("plan_decoder", lambda state: plan_decoder_node(state, model_manager))
    workflow.add_node("execute_model", lambda state: execute_model_node(state, model_manager))
    workflow.add_node("omega_critique", lambda state: omega_critique_node(state, model_manager))
    workflow.add_node("revise_plan", lambda state: revise_plan_node(state, model_manager))

    # --- THE NEW MIND TRANSPLANT ---
    workflow.set_entry_point("social_acuity")
    workflow.add_edge("social_acuity", "determine_pathway")
    workflow.add_edge("determine_pathway", "omega_planner")
    workflow.add_edge("omega_planner", "plan_decoder")
    workflow.add_edge("plan_decoder", "execute_model")
    
    workflow.add_edge("execute_model", "omega_critique")
    workflow.add_conditional_edges(
        "omega_critique",
        should_finish_node,
        {"revise": "revise_plan", "end": END}
    )
    
    # The revision node needs a more complex flow in the future.
    # For now, it will generate a linguistic plan and go back to execution.
    workflow.add_edge("revise_plan", "execute_model")

    logger.info("Cognitive Graph with Omega Planner V2 has been compiled.")
    return workflow.compile()