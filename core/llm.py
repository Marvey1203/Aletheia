# core/llm.py (V2 - Multi-Modal Manager)

import json
from pathlib import Path
from typing import Dict, List, Any
import threading

from loguru import logger
from pydantic import BaseModel

# --- Conditional Imports ---
# These will only be imported if the libraries are installed, preventing crashes.
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# --- Pydantic model for type-safe configuration ---

class ModelDefinition(BaseModel):
    """Represents a single model's configuration in our roster."""
    role: str
    type: str  # 'llm' or 'embedding'
    model_path: str
    hf_link: str
    notes: str

# --- The Model Manager ---

class ModelManager:
    """
    Reads the orchestra_roster from models.json, loads LLM and Embedding models on demand,
    and manages them in memory. This class is thread-safe.
    """
    def __init__(self, config_path: str = "models.json", dummy_mode: bool = False):
        if not dummy_mode and (Llama is None or SentenceTransformer is None):
            logger.error("Missing critical libraries. Please run 'pip install -r requirements.txt'")
            raise ImportError("llama-cpp-python or sentence-transformers is not installed.")

        self.dummy_mode = dummy_mode
        self.roster: Dict[str, ModelDefinition] = self._load_roster(config_path)
        
        # Separate caches for different model types
        self.loaded_llms: Dict[str, Llama] = {}
        self.loaded_embedders: Dict[str, SentenceTransformer] = {}
        
        self.lock = threading.Lock() # Global lock for loading any model
        logger.info(f"ModelManager V2 initialized with {len(self.roster)} models in the roster.")

    def _load_roster(self, config_path: str) -> Dict[str, ModelDefinition]:
        """Loads and validates the models.json configuration file."""
        roster_path = Path(config_path)
        if not roster_path.exists():
            raise FileNotFoundError(f"Could not find model roster: {config_path}")
        
        with open(roster_path, "r") as f:
            data = json.load(f)
            roster = {
                model_data['role']: ModelDefinition(**model_data)
                for model_data in data['orchestra_roster']
            }
            logger.success(f"Successfully loaded and validated {len(roster)} models from {config_path}.")
            return roster

    def get_llm(self, role: str) -> Llama | None:
        """Retrieves an LLM by its role, loading it if necessary."""
        if self.dummy_mode:
            logger.warning(f"Dummy mode active. Skipping LLM load for role: {role}")
            return None

        with self.lock:
            if role in self.loaded_llms:
                return self.loaded_llms[role]

            definition = self.roster.get(role)
            if not definition or definition.type != 'llm':
                raise ValueError(f"Role '{role}' is not a valid 'llm' role in the roster.")

            model_path = Path(definition.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file for role '{role}' not found at {model_path}")

            logger.info(f"Loading LLM for role '{role}' from {model_path}...")
            try:
                llm = Llama(model_path=str(model_path), n_ctx=4096, n_gpu_layers=-1, verbose=False)
                self.loaded_llms[role] = llm
                logger.success(f"Successfully loaded LLM for role '{role}'.")
                return llm
            except Exception as e:
                logger.error(f"Failed to load LLM for role '{role}': {e}")
                raise

    def get_embedding_model(self, role: str) -> SentenceTransformer | None:
        """Retrieves an embedding model by its role, loading it if necessary."""
        if self.dummy_mode:
            logger.warning(f"Dummy mode active. Skipping embedding model load for role: {role}")
            return None

        with self.lock:
            if role in self.loaded_embedders:
                return self.loaded_embedders[role]

            definition = self.roster.get(role)
            if not definition or definition.type != 'embedding':
                raise ValueError(f"Role '{role}' is not a valid 'embedding' role in the roster.")

            # For sentence-transformers, the model_path is the Hugging Face repo ID.
            # The library handles the download and caching automatically.
            model_id = definition.model_path
            logger.info(f"Loading embedding model for role '{role}' from '{model_id}'...")
            try:
                # Run on CPU by default, as it's small and fast
                embedder = SentenceTransformer(model_id, device='cpu')
                self.loaded_embedders[role] = embedder
                logger.success(f"Successfully loaded embedding model for role '{role}'.")
                return embedder
            except Exception as e:
                logger.error(f"Failed to load embedding model for role '{role}': {e}")
                raise

    def generate_text(self, role: str, prompt: str, **kwargs) -> str:
        """Convenience method to generate text with a specific LLM."""
        if self.dummy_mode: return f"Dummy response for '{role}'."
        llm = self.get_llm(role)
        if not llm: return "Error: LLM not found."
        
        try:
            response = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], **kwargs)
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error during text generation with role '{role}': {e}")
            return "Error during generation."

    def create_embedding(self, text: str, role: str = "memory_embedder") -> List[float]:
        """Convenience method to create a vector embedding for a piece of text."""
        if self.dummy_mode: return [0.0] * 384 # Return a dummy vector of the correct size
        embedder = self.get_embedding_model(role)
        if not embedder: return []
        
        try:
            embedding = embedder.encode(text, convert_to_tensor=False, convert_to_numpy=True)
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        except Exception as e:
            logger.error(f"Error during embedding creation with role '{role}': {e}")
            return []