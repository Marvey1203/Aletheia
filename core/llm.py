# core/llm.py
# (Version 1.3 - Thread-Safe with Global Lock)

import json
from pathlib import Path
from typing import Dict, Any
import threading # Import the threading library

from loguru import logger
from llama_cpp import Llama

DEFAULT_MODEL_PATH = "models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"

class LocalLLM:
    """
    A wrapper for the Llama-cpp-python model to handle loading and generation.
    Includes a 'dummy_mode' and is now thread-safe via a global lock.
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, dummy_mode: bool = False):
        self.model_path = Path(model_path)
        self.dummy_mode = dummy_mode
        self.llm = self._load_model()
        # --- NEW: Create a lock for this instance ---
        self.lock = threading.Lock()

    def _load_model(self) -> Llama | None:
        """Loads the GGUF model from the specified path, or returns None in dummy mode."""
        if self.dummy_mode:
            logger.warning("LLM is running in Dummy Mode. No model will be loaded.")
            return None
            
        logger.info(f"Attempting to load LLM from: {self.model_path}")

        if not self.model_path.exists():
            logger.error(f"Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            llm = Llama(
                model_path=str(self.model_path),
                n_ctx=2048,
                n_gpu_layers=0,
                n_threads=4,
                verbose=False
            )
            logger.success("Local LLM loaded successfully into memory.")
            return llm
        except Exception as e:
            logger.error(f"Failed to load the LLM model: {e}")
            raise

    def generate_text(self, prompt: str, temperature: float = 0.2, max_tokens: int = 500) -> str:
        """Generate text response for a given prompt. This method is now thread-safe."""
        if self.dummy_mode or self.llm is None:
            return "This is a simple text response from dummy mode."
        
        # --- NEW: Acquire the lock before using the LLM ---
        with self.lock:
            try:
                response = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=["END_PLAN", "END_EXECUTION", "END_CRITIQUE", "END_TRIAGE"]
                )
                return response['choices'][0]['message']['content'].strip()
            except Exception as e:
                logger.error(f"Error during text generation: {e}")
                return "An error occurred while generating a response."