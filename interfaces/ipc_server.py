# interfaces/ipc_server.py

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
from core.llm import LocalLLM
from core.atp import ATPLoopV2
from core.session import SessionManager # Our new Session Manager

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
    logger.info("Aletheia IPC Server Initializing...")
    logger.info(f"Dummy Mode: {dummy_mode}")
    logger.info("==========================================================")

    try:
        # --- Initialize AI Engine Components ---
        identity_core = IdentityCore()
        memory_galaxy = MemoryGalaxy()
        local_llm = LocalLLM(dummy_mode=dummy_mode)
        atp_loop = ATPLoopV2(
            identity=identity_core,
            memory=memory_galaxy,
            llm=local_llm
        )
        session_manager = SessionManager()
        logger.success("Aletheia Engine is awake and ready.")

        # --- Initialize ZMQ Sockets ---
        context = zmq.Context()
        
        # Command & Control (C2) Socket
        command_socket = context.socket(zmq.REP)
        command_socket.bind(COMMAND_ADDRESS)
        logger.success(f"C2 server is bound and listening on {COMMAND_ADDRESS}")
        
        # Telemetry Publishing Socket
        telemetry_socket = context.socket(zmq.PUB)
        telemetry_socket.bind(TELEMETRY_ADDRESS)
        logger.success(f"Telemetry publisher is bound on {TELEMETRY_ADDRESS}")

        # --- Main Server Loop ---
        while True:
            # 1. Wait for a command on the C2 channel
            command_json = command_socket.recv_json()
            logger.info(f"Received command: {command_json}")

            # 2. Process the command
            if command_json.get("command") == "reason":
                payload = command_json.get("payload", {})
                query = payload.get("query")
                gear_override = payload.get("gear_override")
                
                if not query:
                    error_response = {"status": "error", "message": "No query provided in payload."}
                    command_socket.send_json(error_response)
                    continue

                query_id = f"query_{uuid.uuid4()}"
                
                # 3. Immediately acknowledge the command with a query_id
                command_socket.send_json({"status": "acknowledged", "query_id": query_id})

                # 4. Define the progress callback for telemetry
                def progress_callback(stage: str, result: any):
    
                    # --- FIX START: Make the payload JSON serializable ---
                    payload_data = result
                    if isinstance(result, BaseModel):
                        # If the result is a Pydantic model (like our Reflection object),
                        # convert it to a dictionary first.
                        payload_data = result.model_dump()
                    # --- FIX END ---

                    telemetry_message = {
                        "query_id": query_id,
                        "type": "stage_update",
                        "stage": stage,
                        "payload": payload_data # Use the potentially converted data
                    }
                    # Publish the message on the query_id topic
                    telemetry_socket.send_string(query_id, flags=zmq.SNDMORE)
                    telemetry_socket.send_json(telemetry_message)
                    logger.info(f"Published telemetry for {query_id}: Stage {stage}")


                # 5. Generate context and run the reasoning loop
                context_string = session_manager.generate_trifold_context(query)
                enriched_query = f"{context_string}\n\nQuery: {query}" # Combine context and query

                final_trace = atp_loop.reason(
                    enriched_query, 
                    progress_callback=progress_callback, 
                    gear_override=gear_override
                )
                final_trace.query = query # Overwrite the enriched query with the original for clean history

                # 6. Add the completed trace to our session history
                session_manager.add_trace_to_current_session(final_trace)

                # 7. Publish the final result on the telemetry stream
                final_message = {
                    "query_id": query_id,
                    "type": "final_result",
                    "payload": json.loads(final_trace.model_dump_json()) # Ensure it's a dict
                }
                telemetry_socket.send_string(query_id, flags=zmq.SNDMORE)
                telemetry_socket.send_json(final_message)
                logger.success(f"Published final trace for {query_id}")

            else:
                # Handle unknown commands
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