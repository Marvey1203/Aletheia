# interfaces/ipc_server.py (V3 - Integrated with Cognitive Orchestra)

import sys
import zmq
import json
import uuid
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

# --- Add project root to the Python path for correct module imports ---
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Import Core Aletheia Components ---
from core.identity import IdentityCore
from core.memory import MemoryGalaxy
from core.llm import ModelManager  # UPGRADED: Import ModelManager
from core.atp import ATPLoopV3      # UPGRADED: Import ATPLoopV3
from core.session import SessionManager

# --- Server Configuration ---
COMMAND_ADDRESS = "tcp://127.0.0.1:5555"
TELEMETRY_ADDRESS = "tcp://127.0.0.1:5556"

def main(dummy_mode: bool = False):
    """
    The main function for the Aletheia IPC server.
    Initializes the AI engine and manages the dual-socket communication.
    """
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.info("==========================================================")
    logger.info("Aletheia IPC Server (Cognitive Orchestra Edition) Initializing...")
    logger.info(f"Dummy Mode: {dummy_mode}")
    logger.info("==========================================================")

    try:
        # --- Initialize AI Engine Components ---
        identity_core = IdentityCore()
        memory_galaxy = MemoryGalaxy()
        
        # UPGRADED: Instantiate the ModelManager
        model_manager = ModelManager(dummy_mode=dummy_mode)
        
        # UPGRADED: Instantiate ATPLoopV3 with the ModelManager
        atp_loop = ATPLoopV3(
            identity=identity_core,
            memory=memory_galaxy,
            model_manager=model_manager
        )
        
        session_manager = SessionManager()
        logger.success("Aletheia Engine (V3) is awake and ready.")

        # --- Initialize ZMQ Sockets ---
        context = zmq.Context()
        command_socket = context.socket(zmq.REP)
        command_socket.bind(COMMAND_ADDRESS)
        logger.success(f"C2 server is bound and listening on {COMMAND_ADDRESS}")
        
        telemetry_socket = context.socket(zmq.PUB)
        telemetry_socket.bind(TELEMETRY_ADDRESS)
        logger.success(f"Telemetry publisher is bound on {TELEMETRY_ADDRESS}")

        # --- Main Server Loop ---
        while True:
            command_json = command_socket.recv_json()
            logger.info(f"Received command: {command_json}")

            if command_json.get("command") == "reason":
                payload = command_json.get("payload", {})
                query = payload.get("query")
                gear_override = payload.get("gear_override")
                
                if not query:
                    command_socket.send_json({"status": "error", "message": "No query provided."})
                    continue

                query_id = f"query_{uuid.uuid4()}"
                command_socket.send_json({"status": "acknowledged", "query_id": query_id})

                def progress_callback(stage: str, result: any):
                    payload_data = result
                    if isinstance(result, BaseModel):
                        payload_data = result.model_dump()
                    
                    telemetry_message = {
                        "query_id": query_id,
                        "type": "stage_update",
                        "stage": stage,
                        "payload": payload_data
                    }
                    telemetry_socket.send_string(query_id, flags=zmq.SNDMORE)
                    telemetry_socket.send_json(telemetry_message)
                    logger.info(f"Published telemetry for {query_id}: Stage {stage}")

                context_string = session_manager.generate_trifold_context(query)
                # The V3 loop expects the original query, not the enriched one.
                # It handles its own context and prompting internally.
                final_trace = atp_loop.reason(
                    query, # Pass the original query
                    progress_callback=progress_callback, 
                    gear_override=gear_override
                )
                
                session_manager.add_trace_to_current_session(final_trace)

                final_message = {
                    "query_id": query_id,
                    "type": "final_result",
                    "payload": json.loads(final_trace.model_dump_json())
                }
                telemetry_socket.send_string(query_id, flags=zmq.SNDMORE)
                telemetry_socket.send_json(final_message)
                logger.success(f"Published final trace for {query_id}")

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