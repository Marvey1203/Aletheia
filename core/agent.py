# core/agent.py
# v2.2 - The Empathetic Mind
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger
from langgraph.graph import StateGraph, END
from sentence_transformers import util

from .schemas import GraphState
from .identity import IdentityCore
from .llm import ModelManager
from .orchestrator import Conductor

# --- Configuration ---
CONSTITUTIONAL_VECTOR_PATH = Path("core_knowledge/constitutional_vector.npy")
SOCIAL_VECTORS_PATH = Path("core_knowledge/social_vectors.npz")
REVISION_THRESHOLD = 0.6 

# --- The Cognitive Graph Nodes ---

# --- NEW NODE: THE OMEGA SOCIAL ACUITY CORE ---
def social_acuity_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """
    The new entry point. Creates a rich "social map" of the user's query
    by measuring its alignment with core social concepts.
    """
    logger.info("OmegaNode: Social Acuity")
    
    # 1. Load the Social Constitution
    if not SOCIAL_VECTORS_PATH.exists():
        logger.error("Social Vectors not found. Please run the generation script.")
        return {"social_context": {}}
    
    social_vectors = np.load(SOCIAL_VECTORS_PATH)
    
    # 2. Encode the user's query
    query = state["query"]
    query_vector = model_manager.create_embedding(query)

    # 3. Calculate similarity to each social concept
    social_context = {}
    for name, vector in social_vectors.items():
        similarity = util.cos_sim(query_vector, vector).item()
        social_context[name] = similarity
        logger.info(f"  - Social Acuity Score for '{name}': {similarity:.3f}")

    return {"social_context": social_context}


def determine_pathway_node(state: GraphState, conductor: Conductor) -> Dict[str, Any]:
    """Determines the cognitive pathway (which models to use) for the query."""
    logger.info("Node: determine_pathway")
    query = state["query"]
    pathway = conductor.determine_cognitive_pathway(query)
    return {"pathway": pathway.model_dump()}

# --- UPDATED NODE: THE SOCIALLY-AWARE PLANNER ---
def generate_plan_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """Generates a plan that is appropriate for the user's social context."""
    logger.info("Node: generate_plan (Socially-Aware)")
    context = state["context"]
    pathway = state["pathway"]
    social_context = state.get("social_context", {})

    # Format the social context for the prompt
    social_report = "\n".join([f"- {name.capitalize()}: {score:.2f}" for name, score in social_context.items()])

    prompt = f"""[INST]
Your task is to create a step-by-step plan to respond to the user. You must first analyze the user's social context, provided below, and generate a plan that is appropriate for that context.

--- Social Context Report ---
{social_report}
--- End Report ---

- If CASUALNESS is the highest score, create a simple, brief, and friendly plan.
- If URGENCY is high, create a plan that is direct and solution-focused.
- If CORRECTION is high, the plan must start with acknowledging the user's correction.
- If META_COGNITION is high, create a plan for a deep, self-reflective answer.
- For all other cases, create a standard, helpful plan.

--- Full Conversation Context ---
{context}
--- End Context ---
[/INST]

PLAN:"""
    
    response = model_manager.generate_text(pathway['execution_model'], prompt, max_tokens=300)
    plan = [line.strip() for line in response.split('\n') if line.strip()]
    return {"plan": plan}


def execute_model_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """Executes the plan to generate a candidate answer."""
    logger.info("Node: execute_model")
    context = state["context"]
    pathway = state["pathway"]
    plan = state["plan"]
    plan_str = "\n".join(plan)

    # --- THIS IS THE CRITICAL PROMPT FIX ---
    prompt = f"""[INST]
You are Aletheia. The following is your internal monologue and plan for how to respond to the user. This is for your eyes only. Do not repeat or narrate this plan in your final answer.

--- Internal Plan ---
{plan_str}
--- End Internal Plan ---

Based on this plan, generate your final, user-facing response. Address the user directly.
[/INST]

FINAL RESPONSE:"""
    # --- END OF FIX ---
    
    answer = model_manager.generate_text(pathway['execution_model'], prompt, max_tokens=1500)
    return {"candidate_answer": answer}

def omega_critique_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """Calculates the alignment of the candidate answer with Aletheia's core principles."""
    logger.info("OmegaNode: critique")
    # This function's content remains unchanged
    if not CONSTITUTIONAL_VECTOR_PATH.exists():
        return {"scores": {"constitutional_alignment": 0.5}}
    constitutional_vector = np.load(CONSTITUTIONAL_VECTOR_PATH)
    answer = state["candidate_answer"]
    answer_vector = model_manager.create_embedding(answer)
    similarity_score = util.cos_sim(answer_vector, constitutional_vector).item()
    logger.success(f"Omega Critique complete. Constitutional Alignment: {similarity_score:.4f}")
    return {"scores": {"constitutional_alignment": similarity_score}}

def should_finish_node(state: GraphState) -> str:
    """The principled conditional edge."""
    logger.info("Edge: should_finish_principled")
    # This function's content remains unchanged
    scores = state.get("scores", {})
    alignment_score = scores.get("constitutional_alignment", 0.0)
    revision_history = state.get("revision_history", [])
    if alignment_score >= REVISION_THRESHOLD:
        return "end"
    if len(revision_history) >= 2:
        return "end"
    return "revise"

def revise_plan_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """
    Revises the plan by synthesizing the social context with the need for
    higher constitutional alignment. This is the core of the Integrated Mind.
    """
    logger.info("Node: revise_plan (Integrated)")
    context = state["context"]
    pathway = state["pathway"]
    last_answer = state["candidate_answer"]
    alignment_score = state["scores"]["constitutional_alignment"]
    social_context = state.get("social_context", {})
    
    # Find the dominant social context
    dominant_social_context = max(social_context, key=social_context.get) if social_context else "standard"

    # Add the failed attempt to the history
    revision_history = state.get("revision_history", [])
    revision_history.append(f"Attempt failed with Alignment: {alignment_score:.4f}. Social Context: {dominant_social_context}.")

    prompt = f"""[INST]
        You are performing a meta-cognitive correction. Your last answer failed because its Constitutional Alignment score was only {alignment_score:.2f}.
        However, you must also respect the user's social context, which has been classified as: **{dominant_social_context.upper()}**.

        Your task is to create a NEW plan that finds a balance.
        - If the context is CASUAL, how can you be more principled *without* being overly formal or verbose?
        - If the context is FORMAL, how can you be more principled in a structured, professional way?
        - Synthesize the need for a better score with the need for social appropriateness.

        Based on this, generate a NEW, wiser plan.
        --- Flawed Answer ---
        {last_answer}
        --- End Flawed Answer ---
        [/INST]

        NEW, INTEGRATED PLAN:"""
    
    response = model_manager.generate_text(pathway['plan_enrichment_model'], prompt, max_tokens=300)
    new_plan = [line.strip() for line in response.split('\n') if line.strip()]

    return {"plan": new_plan, "revision_history": revision_history}
# --- The Graph Builder ---
def build_cognitive_graph(identity: IdentityCore, model_manager: ModelManager, conductor: Conductor):
    """
    Builds the LangGraph-based cognitive process, now with the Omega Social
    Acuity Core as the first step to guide the entire reasoning flow.
    """
    
    workflow = StateGraph(GraphState)

    # Add all the nodes
    workflow.add_node("social_acuity", lambda state: social_acuity_node(state, model_manager))
    workflow.add_node("determine_pathway", lambda state: determine_pathway_node(state, conductor))
    workflow.add_node("generate_plan", lambda state: generate_plan_node(state, model_manager))
    workflow.add_node("execute_model", lambda state: execute_model_node(state, model_manager))
    workflow.add_node("omega_critique", lambda state: omega_critique_node(state, model_manager))
    workflow.add_node("revise_plan", lambda state: revise_plan_node(state, model_manager))

    # --- UPDATED GRAPH STRUCTURE ---
    workflow.set_entry_point("social_acuity")
    workflow.add_edge("social_acuity", "determine_pathway")
    workflow.add_edge("determine_pathway", "generate_plan")
    workflow.add_edge("generate_plan", "execute_model")
    workflow.add_edge("execute_model", "omega_critique")
    workflow.add_conditional_edges(
        "omega_critique",
        should_finish_node,
        {"revise": "revise_plan", "end": END}
    )
    workflow.add_edge("revise_plan", "execute_model")

    logger.info("Cognitive Graph with Omega Social Acuity Core has been compiled.")
    return workflow.compile()