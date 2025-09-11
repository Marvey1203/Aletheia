import re
import uuid
from core.atlas import ConceptualAtlas
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger
from langgraph.graph import StateGraph, END
from sentence_transformers import util

from .schemas import DreamGraphState, GraphState, PlanStep, SelfModel
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

# --- NEW OMEGA PLANNER V3 ---
def omega_planner_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """
    Constructs a STRATEGIC, conceptual plan by synthesizing the user's
    social context with the AI's own SelfModel.
    """
    logger.info("OmegaNode: Planner V3 (Strategic)")
    
    query = state["query"]
    social_context = state["social_context"]
    self_model = state.get("self_model") # The wisdom from the subconscious

    # 1. Encode core concepts for this thought process
    query_vec = np.array(model_manager.create_embedding(query), dtype=np.float32)
    plan_action_vectors = {
        name: np.array(model_manager.create_embedding(desc), dtype=np.float32) 
        for name, desc in PLAN_ACTION_LIBRARY.items()
    }

    # 2. The Strategic Synthesis
    # This is the new, wiser logic that replaces the rigid if/elif/else block.
    conceptual_plan = []
    
    # Extract wisdom from the SelfModel
    avg_alignment = float(self_model.statistics.get("average_constitutional_alignment", 1.0)) if self_model else 1.0
    dominant_social_context = max(social_context, key=social_context.get) if social_context else "standard"
    
    logger.info(f"Planner: Dominant social context is '{dominant_social_context}'.")
    logger.info(f"Planner: Consulting SelfModel. Current avg alignment is {avg_alignment:.2f}.")

    # --- The Strategic Rulebook ---

    # Strategy 1: The Meta-Cognition Protocol
    if social_context.get("META_COGNITION", 0.0) > 0.3:
        logger.info("Planner Strategy: Engaging Meta-Cognition Protocol.")
        conceptual_plan.extend([
            PlanStep(action_type=plan_action_vectors["ACKNOWLEDGE"], action_subject=query_vec),
            PlanStep(action_type=plan_action_vectors["ANALYZE"], action_subject=model_manager.create_embedding("my own cognitive architecture and self-model")),
            PlanStep(action_type=plan_action_vectors["RECALL"]), # Recall strategic memory
            PlanStep(action_type=plan_action_vectors["SYNTHESIZE"]),
            PlanStep(action_type=plan_action_vectors["FORMULATE"])
        ])

    # Strategy 2: The Casual Greeting Protocol (with self-awareness)
    elif dominant_social_context == "CASUALNESS":
        logger.info("Planner Strategy: Engaging Casual Greeting Protocol.")
        # If the AI knows it's bad at casual greetings, it adopts a safer, more principled strategy.
        if avg_alignment < 0.4:
            logger.warning(f"Planner: Low alignment score detected ({avg_alignment:.2f}). Overriding casual plan with a more principled approach.")
            conceptual_plan.extend([
                PlanStep(action_type=plan_action_vectors["GREET"]),
                PlanStep(action_type=plan_action_vectors["SYNTHESIZE"]),
                PlanStep(action_type=plan_action_vectors["INQUIRE"])
            ])
        else:
            conceptual_plan.extend([
                PlanStep(action_type=plan_action_vectors["GREET"]),
                PlanStep(action_type=plan_action_vectors["INQUIRE"])
            ])
            
    # Strategy 3: The Default Protocol
    else:
        logger.info("Planner Strategy: Engaging Standard Query Protocol.")
        conceptual_plan.extend([
            PlanStep(action_type=plan_action_vectors["ACKNOWLEDGE"], action_subject=query_vec),
            PlanStep(action_type=plan_action_vectors["ANALYZE"], action_subject=query_vec),
            PlanStep(action_type=plan_action_vectors["RECALL"]),
            PlanStep(action_type=plan_action_vectors["FORMULATE"])
        ])

    logger.success(f"Omega Planner constructed a STRATEGIC plan with {len(conceptual_plan)} steps.")
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

def update_self_model_node(state: GraphState, atlas: ConceptualAtlas) -> Dict[str, Any]:
    """
    The final node. Analyzes the completed trace and saves key strategic
    insights to the strategic_memory collection. This is the core of the
    AI's ability to learn from its own cognitive processes.
    """
    logger.info("Node: update_self_model (Closing the Loop)")

    # 1. Analyze the thought process for key insights
    query = state["query"]
    final_score = state.get("scores", {}).get("constitutional_alignment", 0.0)
    num_revisions = len(state.get("revision_history", []))
    social_context = state.get("social_context", {})
    dominant_social_context = max(social_context, key=social_context.get) if social_context else "standard"

    # 2. Formulate the "meta-trace" or "strategic insight"
    insight_text = (
        f"Strategic Insight: The query was '{query[:50]}...'. "
        f"The dominant social context was '{dominant_social_context}'. "
        f"After {num_revisions} revision(s), a final constitutional alignment of {final_score:.2f} was achieved. "
        f"Revision History: {state.get('revision_history', [])}"
    )
    
    # 3. Save this insight to the strategic memory for future learning
    doc_id = f"strat_{uuid.uuid4()}"
    atlas.strategic_memory_collection.add(
        ids=[doc_id],
        documents=[insight_text],
        metadatas=[{"final_score": final_score, "revisions": num_revisions}]
    )
    
    logger.success(f"Saved strategic insight '{doc_id}' to strategic memory.")
    
    # This node doesn't modify the state further, it just learns from it.
    return {}

# --- The Graph Builder ---
def build_cognitive_graph(identity: IdentityCore, model_manager: ModelManager, conductor: Conductor, atlas: ConceptualAtlas):
    """
    Builds the LangGraph-based cognitive process, now featuring the Omega Planner V2.
    """
    workflow = StateGraph(GraphState)

    # --- THIS IS THE CORRECTED LOGIC ---
    # Step 1: Add ALL nodes to the graph first.
    workflow.add_node("social_acuity", lambda state: social_acuity_node(state, model_manager))
    workflow.add_node("determine_pathway", lambda state: determine_pathway_node(state, conductor))
    workflow.add_node("omega_planner", lambda state: omega_planner_node(state, model_manager))
    workflow.add_node("plan_decoder", lambda state: plan_decoder_node(state, model_manager))
    workflow.add_node("execute_model", lambda state: execute_model_node(state, model_manager))
    workflow.add_node("omega_critique", lambda state: omega_critique_node(state, model_manager))
    workflow.add_node("revise_plan", lambda state: revise_plan_node(state, model_manager))
    workflow.add_node("update_self_model", lambda state: update_self_model_node(state, atlas))

    # Step 2: Now, define the entry point and wire all the edges.
    workflow.set_entry_point("social_acuity")
    workflow.add_edge("social_acuity", "determine_pathway")
    workflow.add_edge("determine_pathway", "omega_planner")
    workflow.add_edge("omega_planner", "plan_decoder")
    workflow.add_edge("plan_decoder", "execute_model")
    workflow.add_edge("execute_model", "omega_critique")
    
    workflow.add_conditional_edges(
        "omega_critique",
        should_finish_node,
        {"revise": "revise_plan", "end": "update_self_model"}
    )
    workflow.add_edge("revise_plan", "omega_planner")
    workflow.add_edge("update_self_model", END)
    # --- END OF CORRECTION ---
    
    logger.info("Cognitive Graph with Strategic Memory Loop has been compiled.")
    return workflow.compile()




# --- THE SUBCONSCIOUS "DREAM ENGINE" ---

def analyze_strategic_memory_node(state: DreamGraphState, atlas: ConceptualAtlas) -> Dict[str, Any]:
    """
    The first 'dream'. This node loads all insights from the strategic memory
    and performs a statistical analysis to update the SelfModel.
    """
    logger.info("DreamNode: Analyzing Strategic Memory")
    
    # 1. Retrieve all strategic insights
    # In a real-world scenario, you'd pull this from the state, but for the first
    # dream, we'll load it directly. A future node could pre-load this.
    try:
        insights = atlas.strategic_memory_collection.get() # Get all items
        if not insights or not insights['metadatas']:
            logger.warning("DreamNode: Strategic memory is empty. Nothing to analyze.")
            return {}
    except Exception as e:
        logger.error(f"DreamNode: Failed to query strategic memory. Error: {e}")
        return {}

    metadatas = insights['metadatas']
    
    # 2. Perform statistical analysis
    if not metadatas:
        return {}
        
    total_thoughts = len(metadatas)
    average_score = np.mean([m.get('final_score', 0.0) for m in metadatas])
    total_revisions = sum(m.get('revisions', 0) for m in metadatas)
    
    new_stats = {
        "total_thoughts_analyzed": total_thoughts,
        "average_constitutional_alignment": f"{average_score:.4f}",
        "total_self_revisions": total_revisions
    }
    
    logger.success(f"DreamNode: Analysis complete. Avg Alignment: {average_score:.4f} across {total_thoughts} thoughts.")

    # 3. Update the SelfModel with the new statistics
    self_model = state.get("self_model", SelfModel())
    self_model.statistics.update(new_stats)
    
    # This dream's output is the updated SelfModel.
    return {"self_model": self_model}


def build_dream_graph(model_manager: ModelManager, atlas: ConceptualAtlas):
    """
    Builds the LangGraph-based subconscious process for Aletheia.
    """
    workflow = StateGraph(DreamGraphState)

    # Add the nodes for the dream process
    workflow.add_node("analyze_memory", lambda state: analyze_strategic_memory_node(state, atlas))

    # Define the graph's structure
    workflow.set_entry_point("analyze_memory")
    workflow.add_edge("analyze_memory", END)

    logger.info("Subconscious 'DreamGraph' has been compiled.")
    return workflow.compile()