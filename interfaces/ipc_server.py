# interfaces/ipc_server.py (V6.0 - The Subconscious Mind)

import sys
import zmq
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
import threading
import time

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
from core.agent import build_cognitive_graph, build_dream_graph
from core.schemas import Trace, Summary, Attempt, Best, Reflection, ModelInfo, GraphState, SelfModel

# --- Global State & Threading Control ---
# This class manages the shared state between the main (conscious) thread
# and the background (subconscious) thread.
class GlobalAletheiaState:
    def __init__(self):
        self.self_model = SelfModel()
        self.is_dreaming = threading.Event()
        self.is_conscious = threading.Event()
        self.shutdown_event = threading.Event()

ALETHEIA_STATE = GlobalAletheiaState()

# --- The "Dream Engine" Background Worker ---
def dream_worker(dream_app, atlas: ConceptualAtlas):
    """
    The background thread that runs the DreamGraph when the AI is idle.
    Its purpose is to perform meta-cognition and update the SelfModel.
    """
    logger.info("DreamWorker: Thread started. Waiting for idle state...")
    while not ALETHEIA_STATE.shutdown_event.is_set():
        # Wait until the is_dreaming event is set by the main thread.
        # This is a non-blocking wait with a timeout.
        dream_on = ALETHEIA_STATE.is_dreaming.wait(timeout=5)
        
        if dream_on:
            logger.warning("DreamWorker: Waking up. The AI is now idle and begins to dream.")
            
            # As long as we're in a dreaming state, run the dream graph periodically.
            while ALETHEIA_STATE.is_dreaming.is_set() and not ALETHEIA_STATE.shutdown_event.is_set():
                initial_dream_state = {"self_model": ALETHEIA_STATE.self_model}
                
                try:
                    # Invoke the simple dream graph
                    final_dream_state = dream_app.invoke(initial_dream_state)
                    
                    # Update the global self_model with the results of the dream
                    if final_dream_state and "self_model" in final_dream_state:
                        ALETHEIA_STATE.self_model = final_dream_state["self_model"]
                        logger.success(f"DreamWorker: Dream complete. SelfModel updated. Stats: {ALETHEIA_STATE.self_model.statistics}")
                    
                    # Wait for a while before the next dream cycle
                    time.sleep(15) # Dream every 15 seconds while idle

                except Exception as e:
                    logger.error(f"DreamWorker: Error during dream cycle: {e}")
                    time.sleep(30) # Wait longer if there's an error
            
            logger.info("DreamWorker: A conscious request was received. Stopping dreams and going to sleep.")

# --- Helper Function for Trace Creation (Now more robust) ---
def _create_trace_from_final_state(final_state: GraphState, identity: IdentityCore) -> Trace:
    """Maps the final state of the cognitive graph to a formal Trace object."""
    logger.info("Mapping final graph state to Omega-aware Trace object...")
    
    trace_id = f"trace_{uuid.uuid4()}"
    timestamp = datetime.now(timezone.utc).isoformat()
    
    scores = final_state.get("scores", {"constitutional_alignment": 0.0})
    total_score = scores.get("constitutional_alignment", 0.0)
    
    # --- THIS IS THE CRITICAL CHANGE ---
    # Use the new 'decoded_answer' key and get the linguistic plan
    final_answer = final_state.get("decoded_answer", "Error: No answer in final state.")
    linguistic_plan = final_state.get("linguistic_plan", [])
    
    attempt = Attempt(
        id=len(final_state.get("revision_history", [])) + 1,
        plan=final_state.get("linguistic_plan", []),
        candidate=final_state.get("decoded_answer", "Error: No answer in final state."),
        scores=scores, 
        total=total_score
    )

    summary = Summary(
        answer=final_answer,
        reasoning=f"Generated via Native Omega Mind. Final Reasoner: {final_state.get('pathway', {}).get('execution_model')}",
        next_action="Awaiting next user query."
    )
    
    best = Best(attempt_id=attempt.id, candidate=final_answer, total=attempt.total)
    # --- END OF CHANGE ---
    
    reflection = Reflection(
        what_worked="The full Omega Core, from planner to executor, ran successfully. The thought was constructed conceptually before being translated to language.",
        what_failed="The reasoning specialists in the Omega Executor are still V1 prototypes and need to be upgraded to Cognitive Operators for deeper reasoning.",
        next_adjustment="Begin R&D on training specialized Cognitive Operators as defined in the Aletheia-Omega memo."
    )
    
    model_info = ModelInfo(
        name=f"Omega Core v1 + {final_state.get('pathway', {}).get('execution_model')} (Decoder)",
        runtime="local",
        temp=0.2
    )

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
    logger.success(f"Successfully created Native Mind Trace {trace.trace_id} from graph state.")
    return trace


# --- Main Server Function ---
def main(dummy_mode: bool = False):
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.info("Aletheia IPC Server (V6.0 - The Subconscious Mind) Initializing...")

    try:
        # --- Component Initialization ---
        identity_core = IdentityCore()
        memory_galaxy = MemoryGalaxy()
        model_manager = ModelManager(identity=identity_core, dummy_mode=dummy_mode)
        atlas = ConceptualAtlas(model_manager=model_manager)
        conductor = Conductor(model_manager=model_manager)
        session_manager = SessionManager(atlas=atlas)
        
        # Build both the conscious and subconscious graphs
        conscious_app = build_cognitive_graph(identity_core, model_manager, conductor, atlas)
        dream_app = build_dream_graph(model_manager, atlas)
        
        logger.success("Aletheia Engine (V6.0) is compiled and ready.")

        # --- Start the Dream Engine background thread ---
        dream_thread = threading.Thread(target=dream_worker, args=(dream_app, atlas), name="DreamWorker")
        dream_thread.daemon = True
        dream_thread.start()
        
        # Start in the "dreaming" state by default
        ALETHEIA_STATE.is_dreaming.set()

        # --- ZMQ setup ---
        context = zmq.Context()
        command_socket = context.socket(zmq.REP)
        command_socket.bind("tcp://127.0.0.1:5555")
        telemetry_socket = context.socket(zmq.PUB)
        telemetry_socket.bind("tcp://127.0.0.1:5556")
        
        logger.info("IPC Sockets are bound. Server is now listening.")

        # --- Main Server Loop (The Conscious Mind) ---
        while True:
            command_json = command_socket.recv_json()

            if command_json.get("command") == "reason":
                # A conscious request has come in. Pause the dreaming.
                ALETHEIA_STATE.is_dreaming.clear()
                ALETHEIA_STATE.is_conscious.set()
                
                payload = command_json.get("payload", {})
                query = payload.get("query")
                if not query:
                    command_socket.send_json({"status": "error", "message": "No query provided."})
                    continue
                
                query_id = f"query_{uuid.uuid4()}"
                command_socket.send_json({"status": "acknowledged", "query_id": query_id})

                context_string = session_manager.generate_trifold_context(query)
                
                # Inject the current SelfModel from the subconscious into the conscious thought
                initial_state = {
                    "query": query, 
                    "context": context_string, 
                    "revision_history": [],
                    "self_model": ALETHEIA_STATE.self_model 
                }

                # Stream telemetry to the UI
                for chunk in conscious_app.stream(initial_state):
                    node_name, node_output = list(chunk.items())[0]
                    # ... (telemetry translator logic here, simplified for now)
                    telemetry_message = { "query_id": query_id, "type": "stage_update", "stage": node_name, "payload": {"status": "completed"} }
                    telemetry_socket.send_string(query_id, flags=zmq.SNDMORE)
                    telemetry_socket.send_json(telemetry_message)
                
                # Get the final state for trace creation
                final_state = conscious_app.invoke(initial_state)
                final_trace = _create_trace_from_final_state(final_state, identity_core)

                # Save and publish the final result
                session_manager.add_trace_to_current_session(final_trace)
                memory_galaxy.save_trace(final_trace)
                atlas.add_trace(final_trace)
                final_message = { "query_id": query_id, "type": "final_result", "payload": json.loads(final_trace.model_dump_json()) }
                telemetry_socket.send_string(query_id, flags=zmq.SNDMORE)
                telemetry_socket.send_json(final_message)
                
                # The conscious task is done. Allow dreaming to resume.
                ALETHEIA_STATE.is_conscious.clear()
                ALETHEIA_STATE.is_dreaming.set()
                
            else:
                command_socket.send_json({"status": "error", "message": "Unknown command"})

    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    except Exception as e:
        logger.error(f"A fatal error occurred in the IPC server: {e}", backtrace=True)
    finally:
        logger.info("Shutting down Aletheia IPC Server.")
        ALETHEIA_STATE.shutdown_event.set()
        if 'dream_thread' in locals() and dream_thread.is_alive():
            dream_thread.join(timeout=2)
        if 'command_socket' in locals():
            command_socket.close()
        if 'telemetry_socket' in locals():
            telemetry_socket.close()
        if 'context' in locals():
            context.term()

if __name__ == "__main__":
    is_dummy = "--dummy" in sys.argv
    main(dummy_mode=is_dummy)