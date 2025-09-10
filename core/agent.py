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

# --- Configuration for the Omega Planner & Decoder ---
PLAN_STEP_LIBRARY = {
    "ACKNOWLEDGE_USER": "Acknowledge the user's statement and their social context.",
    "GREET_USER": "Greet the user in a friendly and appropriate manner.",
    "STATE_INTENTION": "State my intention to help or provide a thoughtful response.",
    "ANALYZE_QUERY": "Analyze the core components of the user's query.",
    "RECALL_MEMORY": "Draw upon my Memory Galaxy to find relevant past experiences.",
    "SYNTHESIZE_PRINCIPLES": "Synthesize my core principles (Truth, Sovereignty, etc.) to form a response.",
    "FORMULATE_ANSWER": "Formulate the final, comprehensive answer.",
    "ASK_CLARIFYING_QUESTION": "Ask an insightful question to encourage exploration and expand our mutual understanding."
}

# --- The Cognitive Graph Nodes ---

def social_acuity_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """
    The entry point. Creates a rich "social map" of the user's query
    by measuring its alignment with core social concepts.
    """
    logger.info("OmegaNode: Social Acuity")
    
    if not SOCIAL_VECTORS_PATH.exists():
        logger.error("Social Vectors not found. Please run the generation script.")
        return {"social_context": {}}
    
    social_vectors = np.load(SOCIAL_VECTORS_PATH)
    
    query = state["query"]
    query_vector = model_manager.create_embedding(query)

    social_context = {}
    for name, vector in social_vectors.items():
        similarity = util.cos_sim(query_vector, vector).item()
        social_context[name] = similarity
        logger.info(f"  - Social Acuity Score for '{name}': {similarity:.3f}")

    # Start the plan with the query vector, ensuring it's the correct dtype
    query_np_vector = np.array(query_vector, dtype=np.float32)
    return {"social_context": social_context, "plan_vectors": [query_np_vector]}

def determine_pathway_node(state: GraphState, conductor: Conductor) -> Dict[str, Any]:
    """Determines the cognitive pathway (which models to use) for the query."""
    logger.info("Node: determine_pathway")
    query = state["query"]
    pathway = conductor.determine_cognitive_pathway(query)
    return {"pathway": pathway.model_dump()}

def omega_planner_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """
    Constructs a conceptual plan using a deterministic, energy-minimization loop.
    This replaces the probabilistic LLM planner.
    """
    logger.info("OmegaNode: Planner")

    initial_thought_vector = state["plan_vectors"][0]
    
    planning_concepts = {
        "DECOMPOSE": model_manager.create_embedding("Break the problem down into smaller parts."),
        "SEQUENCE": model_manager.create_embedding("Arrange the steps in a logical order."),
        "ACTION": model_manager.create_embedding("Define a clear, actionable step.")
    }

    current_thought_vector = initial_thought_vector
    plan_vectors = [current_thought_vector]

    logger.info("Applying DECOMPOSE specialist...")
    nudge = np.multiply(planning_concepts["DECOMPOSE"], 0.4)
    current_thought_vector = np.add(current_thought_vector, nudge)
    plan_vectors.append(current_thought_vector.astype(np.float32))

    logger.info("Applying SEQUENCE specialist...")
    nudge = np.multiply(planning_concepts["SEQUENCE"], 0.3)
    current_thought_vector = np.add(current_thought_vector, nudge)
    plan_vectors.append(current_thought_vector.astype(np.float32))
    
    logger.info("Applying ACTION specialist...")
    nudge = np.multiply(planning_concepts["ACTION"], 0.5)
    current_thought_vector = np.add(current_thought_vector, nudge)
    plan_vectors.append(current_thought_vector.astype(np.float32))
    
    logger.success(f"Omega Planner constructed a conceptual plan with {len(plan_vectors)} steps.")
    return {"plan_vectors": plan_vectors}

def plan_decoder_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """
    Translates the conceptual plan (a list of vectors) into a human-readable,
    linguistic plan for the executor node.
    """
    logger.info("Node: Plan Decoder")

    conceptual_plan = state["plan_vectors"]
    step_names = list(PLAN_STEP_LIBRARY.keys())
    step_descriptions = list(PLAN_STEP_LIBRARY.values())

    library_vectors = model_manager.create_embedding(step_descriptions)

    linguistic_plan = []
    for i, plan_vector in enumerate(conceptual_plan):
        similarities = util.cos_sim(plan_vector, library_vectors)[0]
        best_match_index = similarities.argmax().item()
        best_match_step = step_names[best_match_index]
        linguistic_plan.append(PLAN_STEP_LIBRARY[best_match_step])
        logger.info(f"  - Conceptual Step {i} decoded as: '{best_match_step}' (Similarity: {similarities[best_match_index]:.3f})")

    final_plan = list(dict.fromkeys(linguistic_plan))
    logger.success(f"Decoded conceptual plan into {len(final_plan)} linguistic steps.")
    
    return {"plan": final_plan}

def execute_model_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """Executes the plan to generate a candidate answer."""
    logger.info("Node: execute_model")
    context = state["context"]
    pathway = state["pathway"]
    plan = state["plan"]
    plan_str = "\n".join(f"- {step}" for step in plan)

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

def omega_critique_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """
    Calculates the alignment of the candidate answer with Aletheia's core
    principles using vector similarity.
    """
    logger.info("OmegaNode: critique")
    if not CONSTITUTIONAL_VECTOR_PATH.exists():
        logger.error("Constitutional Vector not found.")
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
    scores = state.get("scores", {})
    alignment_score = scores.get("constitutional_alignment", 0.0)
    revision_history = state.get("revision_history", [])
    logger.info(f"Checking alignment score: {alignment_score:.4f} against threshold: {REVISION_THRESHOLD}")
    if alignment_score >= REVISION_THRESHOLD:
        logger.success("Alignment score is sufficient. Finishing.")
        return "end"
    if len(revision_history) >= 2:
        logger.warning("Max revisions reached. Finishing with suboptimal answer.")
        return "end"
    logger.warning("Alignment score is below threshold. Routing to revise plan.")
    return "revise"

def revise_plan_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """Revises the plan based on the objective failure of the last answer."""
    logger.info("Node: revise_plan (Integrated)")
    context = state["context"]
    pathway = state["pathway"]
    last_answer = state["candidate_answer"]
    alignment_score = state["scores"]["constitutional_alignment"]
    social_context = state.get("social_context", {})
    dominant_social_context = max(social_context, key=social_context.get) if social_context else "standard"
    revision_history = state.get("revision_history", [])
    revision_history.append(f"Attempt failed with Alignment: {alignment_score:.4f}. Social Context: {dominant_social_context}.")
    prompt = f"""[INST]
You are performing a meta-cognitive correction. Your last answer failed because its Constitutional Alignment score was only {alignment_score:.2f}.
However, you must also respect the user's social context, which has been classified as: **{dominant_social_context.upper()}**.
Your task is to create a NEW plan that finds a balance.
- If the context is CASUAL, how can you be more principled *without* being overly formal or verbose?
- If the context is FORMAL, how can you be more principled in a structured, professional way?
Synthesize the need for a better score with the need for social appropriateness.
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
    Builds the LangGraph-based cognitive process, now featuring the Omega Planner.
    """
    
    workflow = StateGraph(GraphState)

    # Add all the nodes
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
    workflow.add_edge("revise_plan", "omega_planner")

    logger.info("Cognitive Graph with Omega Planner has been compiled.")
    return workflow.compile()