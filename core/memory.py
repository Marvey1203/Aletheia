# core/memory.py
# This file defines the MemoryGalaxy, which handles the saving and loading
# of cognitive traces to and from the flat-file system.

from pathlib import Path
from typing import List

from loguru import logger

# Import the Pydantic model for a Trace to ensure type safety
from .schemas import Trace

class MemoryGalaxy:
    """
    Manages the flat-file storage of cognitive traces.
    """
    def __init__(self, storage_path: str = "memory_galaxy"):
        self.storage_path = Path(storage_path)
        self._setup_storage()

    def _setup_storage(self):
        """Ensures the storage directory exists."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Memory Galaxy storage initialized at: {self.storage_path}")

    def save_trace(self, trace: Trace):
        """
        Saves a single Trace object to a JSON file.
        The filename is the trace_id.
        """
        file_path = self.storage_path / f"{trace.trace_id}.json"
        logger.info(f"Saving trace to {file_path}...")
        
        try:
            # Pydantic's model_dump_json is perfect for this. It handles
            # converting the object to a JSON string with proper formatting.
            json_string = trace.model_dump_json(indent=2)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_string)
            logger.success(f"Successfully saved trace: {trace.trace_id}")
        except Exception as e:
            logger.error(f"Failed to save trace {trace.trace_id}: {e}")
            raise

    def load_all_traces(self) -> List[Trace]:
        """
        Loads all trace files from the storage directory into a list of Trace objects.
        """
        logger.info(f"Loading all traces from {self.storage_path}...")
        traces: List[Trace] = []
        for file_path in sorted(self.storage_path.glob("*.json"), reverse=True):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # Pydantic's model_validate_json will read the file, parse it,
                    # and validate it against our schema all in one step.
                    trace = Trace.model_validate_json(f.read())
                    traces.append(trace)
            except Exception as e:
                logger.warning(f"Could not load or validate trace {file_path}: {e}")
        
        logger.success(f"Loaded {len(traces)} traces from the Memory Galaxy.")
        # Sort traces by timestamp for chronological order
        # Note: The sorted glob above already gives us a good default order.
        # A more robust sort would parse the timestamp string.
        traces.sort(key=lambda t: t.timestamp, reverse=True)
        return traces

# Example of how to use it (for standalone testing of this module)
if __name__ == "__main__":
    from datetime import datetime, timezone
    
    # This requires core/schemas.py to exist
    from .schemas import Summary, Best, Reflection, ModelInfo

    # Create a dummy Trace object to test saving
    dummy_summary = Summary(answer="Test", reasoning="Test", next_action="Test")
    dummy_best = Best(attempt_id=1, candidate="Test", total=1.0)
    dummy_reflection = Reflection(what_worked="t", what_failed="t", next_adjustment="t")
    dummy_model_info = ModelInfo(name="test", runtime="test", temp=0.0)
    
    dummy_trace = Trace(
        trace_id="test_trace_001",
        query="This is a test",
        seed_id="eira-001",
        summary=dummy_summary,
        attempts=[],
        best=dummy_best,
        reflection=dummy_reflection,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_info=dummy_model_info,
    )

    memory = MemoryGalaxy()
    memory.save_trace(dummy_trace)
    
    loaded_traces = memory.load_all_traces()
    print(f"\n--- Memory Galaxy Test ---")
    print(f"Found {len(loaded_traces)} trace(s) in the galaxy.")
    if loaded_traces:
        print(f"Latest trace query: '{loaded_traces[0].query}'")
    print("--------------------------")