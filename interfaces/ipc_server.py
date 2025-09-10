# interfaces/ipc_server.py (V5.0 - Omega-Aware Persistence)

import sys
import zmq
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

from loguru import logger

# --- Add project root to the Python path for correct module imports ---
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Import Core Aletheia Components ---
from core.identity import IdentityCore
from core.memory import MemoryGalaxy
from core.llm import ModelManager
from core.orchestrator import Conductor
from core.session import SessionManager
from core.atlas import ConceptualAtlas
from core.agent import build_cognitive_graph
from core.schemas import Trace, Summary, Attempt, Best, Reflection, ModelInfo, GraphState

# --- THIS IS THE UPDATED HELPER FUNCTION ---
def _create_trace_from_final_state(final_state: GraphState, identity: IdentityCore) -> Trace:
    """Maps the final state of the cognitive graph, including Omega scores, to a formal Trace object."""
    logger.info("Mapping final graph state to Omega-aware Trace object...")
    
    trace_id = f"trace_{uuid.uuid4()}"
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # --- THIS IS THE MODIFIED LOGIC ---
    # We now expect a single, powerful score from the OmegaCritiqueNode.
    scores = final_state.get("scores", {"constitutional_alignment": 0.0})
    total_score = scores.get("constitutional_alignment", 0.0)
    # --- END OF MODIFICATION ---
    
    # Create the single, successful attempt from the final state
    attempt = Attempt(
        id=1, # Simplified for now, since we removed the revision loop
        plan=final_state.get("plan", []),
        candidate=final_state.get("candidate_answer", "Error: No answer in final state."),
        scores=scores,
        total=total_score
    )

    # Create the rest of the Trace components
    summary = Summary(
        answer=attempt.candidate,
        reasoning=f"Generated via Cognitive Graph w/ Omega Critique. Reasoner: {final_state.get('pathway', {}).get('execution_model')}",
        next_action="Awaiting next user query."
    )
    
    best = Best(attempt_id=attempt.id, candidate=attempt.candidate, total=attempt.total)
    
    reflection = Reflection(
        what_worked="The OmegaCritiqueNode provided an objective, deterministic score of the answer's alignment with the core constitution.",
        what_failed="The self-correction loop has been temporarily disabled and must be rebuilt using the new, superior Omega score as its guide.",
        next_adjustment="Re-implement the self-correction loop in the Cognitive Graph, using the 'constitutional_alignment' score as the conditional trigger."
    )
    
    model_info = ModelInfo(
        name=f"Orchestra: {final_state.get('pathway', {}).get('execution_model')}",
        runtime="local",
        temp=0.2
    )

    # Assemble the final Trace
    trace = Trace(
        trace_id=trace_id,
        query=final_state["query"],
        seed_id=identity.seed_id,
        summary=summary,
        attempts=[attempt],
        best=best,
        reflection=reflection,
        model_info=model_info,
        timestamp=timestamp
    )
    logger.success(f"Successfully created Omega-aware Trace {trace.trace_id} from graph state.")
    return trace

def main(dummy_mode: bool = False):
    # This main function does not need any changes. The logic is identical to the last working version.
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.info("Aletheia IPC Server (V5.0 - Omega-Aware) Initializing...")

    try:
        identity_core = IdentityCore()
        memory_galaxy = MemoryGalaxy()
        model_manager = ModelManager(identity=identity_core, dummy_mode=dummy_mode)
        atlas = ConceptualAtlas(model_manager=model_manager)
        conductor = Conductor(model_manager=model_manager)
        session_manager = SessionManager(atlas=atlas)
        cognitive_app = build_cognitive_graph(identity_core, model_manager, conductor, atlas)
        logger.success("Aletheia Engine (V5.0) is compiled and ready.")

        context = zmq.Context()
        command_socket = context.socket(zmq.REP)
        command_socket.bind("tcp://127.0.0.1:5555")
        telemetry_socket = context.socket(zmq.PUB)
        telemetry_socket.bind("tcp://127.0.0.1:5556")

        while True:
            command_json = command_socket.recv_json()

            if command_json.get("command") == "reason":
                payload = command_json.get("payload", {})
                query = payload.get("query")
                if not query:
                    command_socket.send_json({"status": "error", "message": "No query provided."})
                    continue
                
                query_id = f"query_{uuid.uuid4()}"
                command_socket.send_json({"status": "acknowledged", "query_id": query_id})

                context_string = session_manager.generate_trifold_context(query)
                initial_state = {"query": query, "context": context_string, "revision_history": []}

                                # --- THIS IS THE CORRECTED LOGIC ---
                final_state = {}
                for chunk in cognitive_app.stream(initial_state):
                    node_name, current_node_output = list(chunk.items())[0]
                    
                    final_state = current_node_output

                    # --- TELEMETRY TRANSLATOR (V2 - NOW AWARE OF SOCIAL ACUITY) ---
                    serializable_output = {}
                    if node_name == "omega_planner":
                        # For the Omega Planner, send a clean summary.
                        num_steps = len(current_node_output.get("plan_vectors", []))
                        serializable_output = {
                            "status": "success",
                            "conceptual_steps_created": num_steps
                        }
                    elif node_name == "social_acuity":
                        # For Social Acuity, send the scores but not the raw vector.
                        # We create a copy and remove the non-serializable part.
                        output_copy = current_node_output.copy()
                        output_copy.pop("plan_vectors", None) # Safely remove the key
                        serializable_output = output_copy
                    else:
                        # For all other nodes, the output is already JSON-safe.
                        serializable_output = current_node_output
                    # --- END OF TRANSLATOR ---

                    telemetry_message = { "query_id": query_id, "type": "stage_update", "stage": node_name, "payload": serializable_output }
                    telemetry_socket.send_string(query_id, flags=zmq.SNDMORE)
                    telemetry_socket.send_json(telemetry_message)
                    logger.info(f"Published telemetry for {query_id}: Node '{node_name}' completed.")
                
                logger.info("Graph streaming complete. Invoking for final state...")
                final_state = cognitive_app.invoke(initial_state)
                
                logger.info(f"Graph invocation complete. Final state keys: {final_state.keys()}")
                
                final_trace = _create_trace_from_final_state(final_state, identity_core)

                session_manager.add_trace_to_current_session(final_trace)
                memory_galaxy.save_trace(final_trace)
                atlas.add_trace(final_trace)

                final_message = {
                    "query_id": query_id,
                    "type": "final_result",
                    "payload": json.loads(final_trace.model_dump_json())
                }
                telemetry_socket.send_string(query_id, flags=zmq.SNDMORE)
                telemetry_socket.send_json(final_message)
                logger.success(f"Published and saved final trace {final_trace.trace_id} for {query_id}")

            else:
                command_socket.send_json({"status": "error", "message": "Unknown command"})

    except Exception as e:
        logger.error(f"A fatal error occurred in the IPC server: {e}", backtrace=True)
    finally:
        logger.info("Shutting down Aletheia IPC Server.")
        command_socket.close()
        telemetry_socket.close()
        context.term()

if __name__ == "__main__":
    is_dummy = "--dummy" in sys.argv
    main(dummy_mode=is_dummy)