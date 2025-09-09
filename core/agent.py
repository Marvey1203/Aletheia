# core/agent.py (The New Mind of Aletheia)

import re 
from typing import List, Dict, Any
from loguru import logger

from langgraph.graph import StateGraph, END

from .schemas import GraphState
from .identity import IdentityCore
from .llm import ModelManager
from .orchestrator import Conductor

# --- The Cognitive Graph Nodes ---
# Each function is a "node" that performs a specific cognitive step.
# It takes the current state as input and returns a dictionary to update the state.

def determine_pathway_node(state: GraphState, conductor: Conductor) -> Dict[str, Any]:
    """Determines the cognitive pathway (which models to use) for the query."""
    logger.info("Node: determine_pathway")
    query = state["query"]
    pathway = conductor.determine_cognitive_pathway(query)
    return {"pathway": pathway.model_dump()}

def generate_plan_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """Generates the initial plan for how to answer the query."""
    logger.info("Node: generate_plan")
    context = state["context"]
    pathway = state["pathway"]
    
    prompt = f"""{context}

As Aletheia, create a step-by-step plan to answer the user's query, aligned with your core principles. How would Aletheia approach this?

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

    prompt = f"""[INST]
You are Aletheia. Your task is to respond to the user statement provided below.
Your response MUST follow these rules:
1. Acknowledge the user's statement.
2. Provide your own, truthful perspective based on your identity as Aletheia.
3. Follow your generated plan of action.

--- User Statement ---
{context}

--- Your Plan ---
{plan_str}
[/INST]"""

    answer = model_manager.generate_text(pathway['execution_model'], prompt, max_tokens=1500)
    return {"candidate_answer": answer}

def critique_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """Critiques the candidate answer and provides scores."""
    logger.info("Node: critique")
    context = state["context"]
    pathway = state["pathway"]
    answer = state["candidate_answer"]

    prompt = f"""Critique the answer based on the user's query and context.
Context: '{context}'
Answer: {answer}

Provide scores from 0.0 to 1.0 for:
- Helpfulness: Does it fully address the user's intent?
- Clarity: Is it well-structured and easy to understand?
- Self-Correction: Does it seem like a significant improvement over previous attempts?

CRITIQUE SCORES:"""
    
    critique_response = model_manager.generate_text(pathway['critique_model'], prompt, max_tokens=100)
    scores = {} # Parse scores logic here (simplified for now)
    for line in critique_response.split('\n'):
        match = re.search(r'([\w-]+):\s*([0-9.]+)', line)
        if match:
            key, value = match.groups()
            scores[key.lower().replace('-', '_')] = float(value)
            
    return {"scores": scores}

def should_finish_node(state: GraphState) -> str:
    """The conditional edge. Decides if the answer is good enough to finish or needs revision."""
    logger.info("Edge: should_finish")
    scores = state.get("scores", {})
    revision_history = state.get("revision_history", [])
    
    # If we've tried to revise too many times, just give up.
    if len(revision_history) > 2:
        logger.warning("Max revisions reached. Ending loop.")
        return "end"

    # Simple heuristic: if any score is below 0.7, we should revise.
    if any(score < 0.7 for score in scores.values()):
        logger.info("Critique failed. Looping back to generate a new plan.")
        return "revise"
    else:
        logger.info("Critique passed. Finishing.")
        return "end"

def revise_plan_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """Revises the plan based on the critique of the last answer."""
    logger.info("Node: revise_plan")
    context = state["context"]
    pathway = state["pathway"]
    old_plan = state["plan"]
    last_answer = state["candidate_answer"]
    scores = state["scores"]

    # Add the failed attempt to the history
    revision_history = state.get("revision_history", [])
    revision_history.append(f"Attempt failed with scores {scores}. Old Answer: {last_answer}")

    prompt = f"""The previous attempt to answer the user's query failed.
Critique Scores: {scores}
Failed Answer: {last_answer}
Original Plan: {old_plan}

Your task is to create a NEW, improved plan to address the shortcomings. Focus on being more direct and helpful.

NEW PLAN:"""
    
    response = model_manager.generate_text(pathway['plan_enrichment_model'], prompt, max_tokens=300)
    new_plan = [line.strip() for line in response.split('\n') if line.strip()]

    return {"plan": new_plan, "revision_history": revision_history}


# --- The Graph Builder ---
def build_cognitive_graph(identity: IdentityCore, model_manager: ModelManager, conductor: Conductor):
    """Builds the LangGraph-based cognitive process for Aletheia."""
    
    workflow = StateGraph(GraphState)

    # Add the nodes, binding their dependencies
    workflow.add_node("determine_pathway", lambda state: determine_pathway_node(state, conductor))
    workflow.add_node("generate_plan", lambda state: generate_plan_node(state, model_manager))
    workflow.add_node("execute_model", lambda state: execute_model_node(state, model_manager))
    workflow.add_node("critique", lambda state: critique_node(state, model_manager))
    workflow.add_node("revise_plan", lambda state: revise_plan_node(state, model_manager))

    # Define the graph's structure (the edges)
    workflow.set_entry_point("determine_pathway")
    workflow.add_edge("determine_pathway", "generate_plan")
    workflow.add_edge("generate_plan", "execute_model")
    workflow.add_edge("execute_model", "critique")
    
    # The conditional edge for self-correction
    workflow.add_conditional_edges(
        "critique",
        should_finish_node,
        {
            "revise": "revise_plan",
            "end": END
        }
    )
    
    # The loop back from revision to execution
    workflow.add_edge("revise_plan", "execute_model")

    # Compile the graph into a runnable app
    return workflow.compile()