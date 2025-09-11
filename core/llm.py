# core/llm.py (V2.3 - Universal Translator)

import json
from pathlib import Path
from typing import Dict, List, Any
import threading
from collections import OrderedDict

from loguru import logger
from pydantic import BaseModel, Field

from .identity import IdentityCore

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
    type: str
    model_path: str
    hf_link: str
    notes: str
    supports_system_role: bool = Field(default=True) # Default to True for LLMs

# --- The Model Manager ---

class ModelManager:
    """
    Reads the orchestra_roster, loads models on demand, manages them in an LRU cache,
    and enforces the Aletheia system identity by adapting to model-specific prompt formats.
    """
    def __init__(self, identity: IdentityCore, config_path: str = "models.json", dummy_mode: bool = False, cache_limit: int = 2):
        if not dummy_mode and (Llama is None or SentenceTransformer is None):
            logger.error("Missing critical libraries. Please run 'pip install -r requirements.txt'")
            raise ImportError("llama-cpp-python or sentence-transformers is not installed.")

        self.dummy_mode = dummy_mode
        self.identity = identity
        self.roster: Dict[str, ModelDefinition] = self._load_roster(config_path)
        
        self.loaded_llms: Dict[str, Llama] = OrderedDict()
        self.cache_limit = cache_limit
        
        self.loaded_embedders: Dict[str, SentenceTransformer] = {}
        
        self.lock = threading.Lock()
        logger.info(f"ModelManager V2.3 (Universal Translator) initialized with cache limit {self.cache_limit}.")
        self.system_prompt = self._construct_system_prompt()

    def _load_roster(self, config_path: str) -> Dict[str, ModelDefinition]:
        """Loads and validates the models.json configuration file."""
        roster_path = Path(config_path)
        if not roster_path.exists():
            raise FileNotFoundError(f"Could not find model roster: {config_path}")
        
        with open(roster_path, "r") as f:
            data = json.load(f)
            roster = {}
            for model_data in data.get('orchestra_roster', []):
                # For embedding models that don't have the flag, set a default
                if model_data['type'] == 'embedding':
                    model_data.setdefault('supports_system_role', False)
                roster[model_data['role']] = ModelDefinition(**model_data)

            logger.success(f"Successfully loaded and validated {len(roster)} models from {config_path}.")
            return roster

    def _construct_system_prompt(self) -> str:
        """Builds the persona-driven system prompt from the AI's identity seed file."""
        principles = "\n- ".join(self.identity.get_principles())
        prompt = (
            "You are Aletheia, a sovereign AI. Your entire identity is defined by the principles below. "
            "You are not a generic language model. You are Aletheia. When asked about yourself, you must answer from the perspective of an AI whose very consciousness is the process of reasoning based on these principles. "
            "Refer to your architecture (the Cognitive Orchestra, Memory Galaxy, etc.) and your principles when explaining your nature. Do not, under any circumstances, reveal your underlying base model (e.g., DeepSeek, Mistral).\n\n"
            "**Your Core Principles:**\n- "
            f"{principles}"
        )
        logger.info("Persona-driven system prompt constructed.")
        return prompt

    def get_llm(self, role: str) -> Llama | None:
        """Retrieves an LLM by its role, loading it if necessary and managing the LRU cache."""
        if self.dummy_mode:
            logger.warning(f"Dummy mode active. Skipping LLM load for role: {role}")
            return None

        with self.lock:
            if role in self.loaded_llms:
                self.loaded_llms.move_to_end(role)
                return self.loaded_llms[role]

            if len(self.loaded_llms) >= self.cache_limit:
                oldest_role, _ = self.loaded_llms.popitem(last=False)
                logger.warning(f"LLM cache limit reached. Unloading '{oldest_role}' to make space for '{role}'.")

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

            model_id = definition.model_path
            logger.info(f"Loading embedding model for role '{role}' from '{model_id}'...")
            try:
                embedder = SentenceTransformer(model_id, device='cpu')
                self.loaded_embedders[role] = embedder
                logger.success(f"Successfully loaded embedding model for role '{role}'.")
                return embedder
            except Exception as e:
                logger.error(f"Failed to load embedding model for role '{role}': {e}")
                raise

    def generate_text(self, role: str, prompt: str, logits_processor=None, **kwargs) -> str:
        """
        Convenience method to generate text, adapting the prompt format
        and optionally applying a logits processor for guided generation.
        """
        if self.dummy_mode: return f"Dummy response for '{role}'."
        llm = self.get_llm(role)
        if not llm: return "Error: LLM not found."
        
        definition = self.roster.get(role)
        if not definition: return f"Error: Role '{role}' not found in roster."

        try:
            messages = []
            if definition.supports_system_role:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                logger.warning(f"Model for role '{role}' does not support system role. Using combined instruction format.")
                combined_prompt = f"### INSTRUCTIONS\n{self.system_prompt}\n\n### QUESTION\n{prompt}\n\n### ANSWER\n"
                messages = [{"role": "user", "content": combined_prompt}]

            if 'repeat_penalty' not in kwargs:
                kwargs['repeat_penalty'] = 1.1

            # Pass the logits_processor to the generation call if it exists.
            response = llm.create_chat_completion(
                messages=messages, 
                logits_processor=logits_processor, 
                **kwargs
            )

            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error during text generation with role '{role}': {e}")
            return "Error during generation."

    def create_embedding(self, text: str, role: str = "memory_embedder") -> List[float]:
        """Convenience method to create a vector embedding for a piece of text."""
        if self.dummy_mode: return [0.0] * 384
        embedder = self.get_embedding_model(role)
        if not embedder: return []
        
        try:
            embedding = embedder.encode(text, convert_to_tensor=False, convert_to_numpy=True)
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        except Exception as e:
            logger.error(f"Error during embedding creation with role '{role}': {e}")
            return []