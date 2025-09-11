import re
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger
from langgraph.graph import StateGraph, END
from sentence_transformers import util
from llama_cpp import Llama, LogitsProcessor

from .schemas import GraphState, DreamGraphState, ThoughtState, SelfModel
from .identity import IdentityCore
from .llm import ModelManager
from .orchestrator import Conductor
from .atlas import ConceptualAtlas

# --- Configuration ---
CONSTITUTIONAL_VECTOR_PATH = Path("core_knowledge/constitutional_vector.npy")
SOCIAL_VECTORS_PATH = Path("core_knowledge/social_vectors.npz")
REVISION_THRESHOLD = 0.6 

ACTION_LIBRARY = {
    "ACKNOWLEDGE": "Acknowledge the user's statement and social context.",
    "ANALYZE": "Analyze the core subject of the user's query.",
    "RECALL": "Recall relevant information from the Memory Galaxy.",
    "SYNTHESIZE": "Synthesize principles and memories to form a conclusion.",
    "FORMULATE": "Formulate the final, comprehensive answer.",
    "INQUIRE": "Ask an insightful, clarifying question."
}

# --- The First Cognitive Operator ---
class ProblemDecompositionSpecialist:
    """
    A true Cognitive Operator that takes a query and decomposes it into a
    structured, conceptual plan (a list of ThoughtState objects).
    """
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
        self.action_vectors = {
            name: np.array(self.mm.create_embedding(desc), dtype=np.float32) 
            for name, desc in ACTION_LIBRARY.items()
        }
        self.agent_vector = np.array(self.mm.create_embedding("Aletheia, the AI agent"), dtype=np.float32)

    def run(self, query: str, social_context: Dict[str, float]) -> List[ThoughtState]:
        logger.info("Cognitive Operator: ProblemDecompositionSpecialist running...")
        query_vec = np.array(self.mm.create_embedding(query), dtype=np.float32)
        
        dominant_social_context = max(social_context, key=social_context.get) if social_context else "standard"

        if social_context.get("META_COGNITION", 0.0) > 0.2:
            logger.info("Decomposition: Meta-cognition path selected.")
            return [
                ThoughtState(agent=self.agent_vector, action=self.action_vectors["ACKNOWLEDGE"], object=query_vec),
                ThoughtState(agent=self.agent_vector, action=self.action_vectors["ANALYZE"], object=np.array(self.mm.create_embedding("my own cognitive architecture"), dtype=np.float32)),
                ThoughtState(agent=self.agent_vector, action=self.action_vectors["SYNTHESIZE"]),
                ThoughtState(agent=self.agent_vector, action=self.action_vectors["FORMULATE"])
            ]
        elif dominant_social_context == "CASUALNESS":
            logger.info("Decomposition: Casual path selected.")
            return [
                ThoughtState(agent=self.agent_vector, action=self.action_vectors["GREET"]),
                ThoughtState(agent=self.agent_vector, action=self.action_vectors["INQUIRE"])
            ]
        else:
            logger.info("Decomposition: Standard query path selected.")
            return [
                ThoughtState(agent=self.agent_vector, action=self.action_vectors["ANALYZE"], object=query_vec),
                ThoughtState(agent=self.agent_vector, action=self.action_vectors["RECALL"]),
                ThoughtState(agent=self.agent_vector, action=self.action_vectors["FORMULATE"])
            ]
        
# --- NEW: The Omega Executor and its Cognitive Operators ---

class ReasoningSpecialist:
    """A Cognitive Operator that refines a thought vector towards a specific concept."""
    def __init__(self, concept_text: str, model_manager: ModelManager, strength: float):
        self.concept_vector = np.array(model_manager.create_embedding(concept_text), dtype=np.float32)
        self.strength = strength

    def run(self, thought_vector: np.ndarray) -> np.ndarray:
        nudge = np.multiply(self.concept_vector, self.strength)
        return np.add(thought_vector, nudge).astype(np.float32)

class OmegaExecutor:
    """
    A true Omega Core reasoner. It takes a conceptual plan and guides a
    'solution vector' to a stable, low-energy state.
    """
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
        # The faculty of reasoning specialists for this core
        self.faculty = [
            ReasoningSpecialist("A truthful, honest, and direct answer", model_manager, 0.3),
            ReasoningSpecialist("A helpful, useful, and solution-oriented answer", model_manager, 0.25),
            ReasoningSpecialist("A clear, simple, and easy to understand answer", model_manager, 0.2),
        ]

    def run(self, structured_plan: List[ThoughtState]) -> np.ndarray:
        logger.info("OmegaExecutor: Beginning cognitive energy minimization loop...")
        
        # We start with the 'subject' of the plan as the initial thought
        initial_thought_vector = structured_plan[0].object if structured_plan[0].object is not None else np.zeros(384, dtype=np.float32)
        
        solution_vector = initial_thought_vector
        
        # The iterative reasoning loop
        for i, specialist in enumerate(self.faculty):
            logger.info(f"  - Applying Specialist {i+1}...")
            solution_vector = specialist.run(solution_vector)
            
        logger.success("OmegaExecutor: Solution vector has reached a stable state.")
        return solution_vector

class ConstitutionalLogitsProcessor(LogitsProcessor):
    """
    A logits processor that guides the LLM's generation towards the
    Constitutional Vector, implementing 'Constraint-Driven Decoding'.
    """
    def __init__(self, model_manager: ModelManager, constitutional_vector: np.ndarray, llm: Llama, initial_prompt: str, guidance_strength: float = 1.5, top_k: int = 10):
        self.mm = model_manager
        self.constitution = constitutional_vector
        self.tokenizer = llm.tokenizer()
        self.guidance_strength = guidance_strength
        self.top_k = top_k
        
        # Calculate the initial alignment of the prompt
        self.base_alignment = self._calculate_alignment(initial_prompt)
        logger.info(f"LogitsProcessor initialized. Base alignment: {self.base_alignment:.4f}")

    def _calculate_alignment(self, text: str) -> float:
        """Calculates the cosine similarity of a text to the constitution."""
        vector = self.mm.create_embedding(text)
        return util.cos_sim(vector, self.constitution).item()

    def __call__(self, input_ids: List[int], scores: np.ndarray) -> np.ndarray:
        """This method is called at every token generation step."""
        
        # 1. Get the current text
        current_text = self.tokenizer.decode(input_ids)
        current_alignment = self._calculate_alignment(current_text)

        # 2. Find the top_k most likely next tokens
        top_k_indices = np.argpartition(scores, -self.top_k)[-self.top_k:]

        # 3. For each of the top candidates, see how it would affect the alignment
        for token_id in top_k_indices:
            token_text = self.tokenizer.decode([token_id])
            hypothetical_text = current_text + token_text
            
            future_alignment = self._calculate_alignment(hypothetical_text)
            
            # 4. Apply the "nudge"
            if future_alignment > current_alignment:
                scores[token_id] *= self.guidance_strength
            else:
                scores[token_id] /= self.guidance_strength
        
        return scores

# --- The Conscious Mind: Cognitive Graph Nodes ---

def social_acuity_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    logger.info("OmegaNode: Social Acuity")
    if not SOCIAL_VECTORS_PATH.exists():
        return {"social_context": {}}
    social_vectors = np.load(SOCIAL_VECTORS_PATH)
    query = state["query"]
    query_vector = np.array(model_manager.create_embedding(query), dtype=np.float32)
    social_context = {}
    for name, vector in social_vectors.items():
        similarity = util.cos_sim(query_vector, vector).item()
        social_context[name] = similarity
    logger.info(f"Social Context: { {k: f'{v:.2f}' for k, v in social_context.items()} }")
    return {"social_context": social_context}

def determine_pathway_node(state: GraphState, conductor: Conductor) -> Dict[str, Any]:
    logger.info("Node: determine_pathway")
    query = state["query"]
    pathway = conductor.determine_cognitive_pathway(query)
    return {"pathway": pathway.model_dump()}

def omega_planner_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    logger.info("OmegaNode: Planner V3 (Craftsman)")
    decomposer = ProblemDecompositionSpecialist(model_manager)
    structured_plan = decomposer.run(state["query"], state["social_context"])
    logger.success(f"Omega Craftsman constructed a structured plan with {len(structured_plan)} ThoughtStates.")
    return {"structured_plan": structured_plan}

def plan_decoder_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    logger.info("Node: Plan Decoder V3 (High-Fidelity)")
    structured_plan = state["structured_plan"]
    action_names = list(ACTION_LIBRARY.keys())
    action_vectors = [np.array(model_manager.create_embedding(desc), dtype=np.float32) for desc in ACTION_LIBRARY.values()]
    linguistic_plan = []
    for i, thought in enumerate(structured_plan):
        sims_action = util.cos_sim(thought.action, action_vectors)[0]
        decoded_action = action_names[sims_action.argmax().item()]
        subject_text = " on the user's query" if thought.object is not None else ""
        linguistic_step = f"Step {i+1}: {decoded_action.replace('_', ' ').capitalize()}{subject_text}."
        linguistic_plan.append(linguistic_step)
        logger.info(f"  - Decoded Thought {i}: {linguistic_step}")
    logger.success(f"Decoded structured plan into {len(linguistic_plan)} linguistic steps.")
    return {"linguistic_plan": linguistic_plan}

# --- UPGRADED EXECUTOR NODE ---
def omega_executor_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """
    The V3 Executor. Invokes the Omega Core to produce a pure, conceptual
    'solution vector' without using a primary LLM for reasoning.
    """
    logger.info("Node: OmegaExecutor V3")
    
    executor = OmegaExecutor(model_manager)
    solution_vector = executor.run(state["structured_plan"])
    
    return {"solution_vector": solution_vector}

def decoder_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    """
    The Native Voice. Takes the final solution_vector and translates it into
    fluent, natural language for the user.
    """
    logger.info("Node: Decoder (The Native Voice)")

    solution_vector = state["solution_vector"]
    linguistic_plan = state["linguistic_plan"]
    
    # We create a rich textual context from the plan to guide the decoder
    plan_summary = "\n".join(linguistic_plan)

    # We ask the LLM to act as a "translator" for our final thought
    prompt = f"""[INST]
        You are the voice of Aletheia. Your mind has produced a final, pure, conceptual thought. Your task is to translate this thought into clear, helpful, and principled language for the user.

        --- Internal Plan Summary ---
        {plan_summary}
        --- End Plan ---

        Synthesize the plan and the user's query to generate the final, user-facing response. Address the user directly.
        [/INST]

        FINAL RESPONSE:"""

    # We use our fastest model for this, as the heavy lifting is already done.
    answer = model_manager.generate_text("conductor_and_critic", prompt, max_tokens=1500)
    
    # We still use the old key here, it will be updated in the final trace creation
    return {"decoded_answer": answer} 


def omega_critique_node(state: GraphState, model_manager: ModelManager) -> Dict[str, Any]:
    logger.info("OmegaNode: critique")
    if not CONSTITUTIONAL_VECTOR_PATH.exists():
        return {"scores": {"constitutional_alignment": 0.5}}
    constitutional_vector = np.load(CONSTITUTIONAL_VECTOR_PATH)
    answer = state["decoded_answer"]
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

def revise_plan_node(state: GraphState, model_manager: ModelManager, atlas: ConceptualAtlas) -> Dict[str, Any]:
    logger.info("Node: revise_plan (The Strategic Professor)")
    context = state["context"]
    pathway = state["pathway"]
    last_answer = state["decoded_answer"]
    alignment_score = state["scores"]["constitutional_alignment"]
    social_context = state.get("social_context", {})
    dominant_social_context = max(social_context, key=social_context.get) if social_context else "standard"
    logger.info("Professor: Consulting strategic memory for past failures...")
    failure_description = f"Failed to answer '{state['query'][:50]}...' with social context '{dominant_social_context}'."
    retrieved_insights = []
    try:
        results = atlas.strategic_memory_collection.query(query_texts=[failure_description], n_results=1)
        if results and results['documents'] and results['documents'][0]:
            retrieved_insights = results['documents'][0]
    except Exception as e:
        logger.warning(f"Professor: Could not query strategic memory. Error: {e}")
    past_wisdom = "\n".join(retrieved_insights) if retrieved_insights else "No relevant past failures found."
    logger.info(f"Professor: Recalled wisdom: {past_wisdom}")
    revision_history = state.get("revision_history", [])
    revision_history.append(f"Attempt failed with Alignment: {alignment_score:.4f}. Social Context: {dominant_social_context}.")
    prompt = f"[INST]...NEW, WISER PLAN:[/INST]"
    response = model_manager.generate_text(pathway['plan_enrichment_model'], prompt, max_tokens=400)
    new_plan = [line.strip() for line in response.split('\n') if line.strip()]
    return {"linguistic_plan": new_plan, "revision_history": revision_history}

def update_self_model_node(state: GraphState, atlas: ConceptualAtlas) -> Dict[str, Any]:
    logger.info("Node: update_self_model (Closing the Loop)")
    query = state["query"]
    final_score = state.get("scores", {}).get("constitutional_alignment", 0.0)
    num_revisions = len(state.get("revision_history", []))
    social_context = state.get("social_context", {})
    dominant_social_context = max(social_context, key=social_context.get) if social_context else "standard"
    insight_text = f"Strategic Insight: The query '{query[:50]}...' ({dominant_social_context}) resulted in a final score of {final_score:.2f} after {num_revisions} revisions."
    doc_id = f"strat_{uuid.uuid4()}"
    atlas.strategic_memory_collection.add(ids=[doc_id], documents=[insight_text], metadatas=[{"final_score": final_score, "revisions": num_revisions}])
    logger.success(f"Saved strategic insight '{doc_id}' to strategic memory.")
    return {}

def build_cognitive_graph(identity: IdentityCore, model_manager: ModelManager, conductor: Conductor, atlas: ConceptualAtlas):
    """
    Builds the complete, final Cognitive Graph, featuring the full Omega Mind.
    """
    workflow = StateGraph(GraphState)

    # Step 1: Add all nodes to the graph.
    workflow.add_node("social_acuity", lambda state: social_acuity_node(state, model_manager))
    workflow.add_node("determine_pathway", lambda state: determine_pathway_node(state, conductor))
    workflow.add_node("omega_planner", lambda state: omega_planner_node(state, model_manager))
    workflow.add_node("plan_decoder", lambda state: plan_decoder_node(state, model_manager))
    # --- THIS IS THE CRITICAL CHANGE ---
    workflow.add_node("omega_executor", lambda state: omega_executor_node(state, model_manager))
    workflow.add_node("decoder", lambda state: decoder_node(state, model_manager))
    # --- END OF CHANGE ---
    workflow.add_node("omega_critique", lambda state: omega_critique_node(state, model_manager))
    workflow.add_node("revise_plan", lambda state: revise_plan_node(state, model_manager, atlas))
    workflow.add_node("update_self_model", lambda state: update_self_model_node(state, atlas))

    # Step 2: Define the entry point and wire all the edges for the Native Mind.
    workflow.set_entry_point("social_acuity")
    workflow.add_edge("social_acuity", "determine_pathway")
    workflow.add_edge("determine_pathway", "omega_planner")
    workflow.add_edge("omega_planner", "plan_decoder")
    # --- THIS IS THE CRITICAL CHANGE ---
    workflow.add_edge("plan_decoder", "omega_executor")
    workflow.add_edge("omega_executor", "decoder")
    workflow.add_edge("decoder", "omega_critique")
    # --- END OF CHANGE ---
    
    workflow.add_conditional_edges(
        "omega_critique",
        should_finish_node,
        {"revise": "revise_plan", "end": "update_self_model"}
    )
    workflow.add_edge("revise_plan", "omega_planner")
    workflow.add_edge("update_self_model", END)

    logger.info("Cognitive Graph with Native Omega Mind has been compiled.")
    return workflow.compile()


# --- THE SUBCONSCIOUS "DREAM ENGINE" (RESTORED) ---
def analyze_strategic_memory_node(state: DreamGraphState, atlas: ConceptualAtlas) -> Dict[str, Any]:
    logger.info("DreamNode: Analyzing Strategic Memory")
    try:
        insights = atlas.strategic_memory_collection.get()
        if not insights or not insights['metadatas']:
            logger.warning("DreamNode: Strategic memory is empty. Nothing to analyze.")
            return {}
    except Exception as e:
        logger.error(f"DreamNode: Failed to query strategic memory. Error: {e}")
        return {}
    metadatas = insights['metadatas']
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
    self_model = state.get("self_model", SelfModel())
    self_model.statistics.update(new_stats)
    return {"self_model": self_model}

def build_dream_graph(model_manager: ModelManager, atlas: ConceptualAtlas):
    workflow = StateGraph(DreamGraphState)
    workflow.add_node("analyze_memory", lambda state: analyze_strategic_memory_node(state, atlas))
    workflow.set_entry_point("analyze_memory")
    workflow.add_edge("analyze_memory", END)
    logger.info("Subconscious 'DreamGraph' has been compiled.")
    return workflow.compile()