# FILE: interfaces/ipc_server.py (DEBUGGING VERSION)

print("Step 1: Starting script...")
import sys
from pathlib import Path
import zmq
from loguru import logger
print("Step 2: Basic imports OK.")

# --- Add project root to the Python path for correct module imports ---
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
print("Step 3: Project root path added.")

# --- Import Core Aletheia Components ---
print("Step 4: Attempting to import IdentityCore...")
from core.identity import IdentityCore
print("Step 5: Attempting to import MemoryGalaxy...")
from core.memory import MemoryGalaxy
print("Step 6: Attempting to import LocalLLM...")
from core.llm import LocalLLM
print("Step 7: Attempting to import ATPLoopV2...")
from core.atp import ATPLoopV2
print("Step 8: All core imports OK.")

# --- Server Configuration ---
IPC_ADDRESS = "tcp://127.0.0.1:5555"

def main(dummy_mode: bool = False):
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.info("==========================================================")
    logger.info("Aletheia IPC Server Initializing...")
    # ... (rest of the file is the same)
    logger.info(f"IPC Address: {IPC_ADDRESS}")
    logger.info(f"Dummy Mode: {dummy_mode}")
    logger.info("==========================================================")

    try:
        identity_core = IdentityCore()
        memory_galaxy = MemoryGalaxy()
        local_llm = LocalLLM(dummy_mode=dummy_mode)
        atp_loop = ATPLoopV2(
            identity=identity_core,
            memory=memory_galaxy,
            llm=local_llm
        )
        logger.success("Aletheia Engine is awake and ready.")

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(IPC_ADDRESS)
        logger.success(f"ZMQ server is bound and listening on {IPC_ADDRESS}")

        while True:
            query_string = socket.recv_string()
            logger.info(f"Received query from client: \"{query_string}\"")

            final_trace = atp_loop.reason(query_string)
            trace_json_string = final_trace.model_dump_json()

            socket.send_string(trace_json_string)
            logger.success(f"Sent trace response for ID: {final_trace.trace_id}")

    except Exception as e:
        logger.error(f"A fatal error occurred in the IPC server: {e}", backtrace=True)
    finally:
        logger.info("Shutting down Aletheia IPC Server.")
        socket.close()
        context.term()

if __name__ == "__main__":
    is_dummy = "--dummy" in sys.argv
    main(dummy_mode=is_dummy)