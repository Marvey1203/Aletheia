# core/llm.py

import json
from pathlib import Path
from typing import Dict, List
import threading

from loguru import logger
from llama_cpp import Llama
from pydantic import BaseModel

# --- Pydantic model for type-safe configuration ---

class ModelDefinition(BaseModel):
    """Represents a single model's configuration in our roster."""
    role: str
    model_path: str
    hf_link: str
    notes: str

# --- The Model Manager ---

class ModelManager:
    """
    Reads the orchestra_roster from models.json, loads models on demand,
    and manages them in memory. This class is thread-safe.
    """
    def __init__(self, config_path: str = "models.json", dummy_mode: bool = False):
        self.dummy_mode = dummy_mode
        self.roster: Dict[str, ModelDefinition] = self._load_roster(config_path)
        self.loaded_models: Dict[str, Llama] = {}
        self.lock = threading.Lock() # To prevent race conditions when loading models
        logger.info(f"ModelManager initialized with {len(self.roster)} models in the roster.")

    def _load_roster(self, config_path: str) -> Dict[str, ModelDefinition]:
        """Loads and validates the models.json configuration file."""
        roster_path = Path(config_path)
        if not roster_path.exists():
            logger.error(f"Model roster not found at {config_path}")
            raise FileNotFoundError(f"Could not find model roster: {config_path}")
        
        with open(roster_path, "r") as f:
            data = json.load(f)
            # Use a dictionary comprehension for efficient, role-keyed access
            roster = {
                model_data['role']: ModelDefinition(**model_data)
                for model_data in data['orchestra_roster']
            }
            logger.success(f"Successfully loaded and validated {len(roster)} models from {config_path}.")
            return roster

    def get_model(self, role: str) -> Llama | None:
        """
        Retrieves a model by its role. Loads the model into memory if it hasn't been already.
        This method is thread-safe.
        """
        if self.dummy_mode:
            logger.warning(f"Dummy mode is active. Skipping model load for role: {role}")
            return None

        # First, check if the model is already loaded without locking
        if role in self.loaded_models:
            return self.loaded_models[role]

        # If not loaded, acquire the lock to prevent multiple threads loading the same model
        with self.lock:
            # Double-check if another thread loaded it while we were waiting for the lock
            if role in self.loaded_models:
                return self.loaded_models[role]

            if role not in self.roster:
                logger.error(f"Model role '{role}' not found in the roster.")
                raise ValueError(f"Unknown model role: {role}")

            definition = self.roster[role]
            model_path = Path(definition.model_path)

            if not model_path.exists():
                logger.error(f"Model file not found for role '{role}' at {model_path}")
                raise FileNotFoundError(f"Model file not found for role '{role}': {model_path}")

            logger.info(f"Loading model for role '{role}' from {model_path}...")
            
            try:
                # TODO: In the future, GPU layers and other settings can come from models.json
                llm = Llama(
                    model_path=str(model_path),
                    n_ctx=4096, # Increased context size
                    n_gpu_layers=-1, # Load all possible layers to GPU
                    n_threads=8,
                    verbose=False
                )
                self.loaded_models[role] = llm
                logger.success(f"Successfully loaded model for role '{role}'.")
                return llm
            except Exception as e:
                logger.error(f"Failed to load the LLM for role '{role}': {e}")
                raise

    def generate_text(self, role: str, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        """A convenience method to get a model and generate text in one call."""
        if self.dummy_mode:
            return f"This is a dummy response for a '{role}' model."

        llm = self.get_model(role)
        if llm is None:
            return "Error: Could not retrieve model."
            
        try:
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["<|end_of_text|>", "<|endoftext|>", "END_"] # Add common stop tokens
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error during text generation with role '{role}': {e}")
            return "An error occurred while generating a response."