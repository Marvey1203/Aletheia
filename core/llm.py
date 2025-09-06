# core/llm.py
# Enhanced LLM class with performance optimizations

import json
import signal
from pathlib import Path
from typing import Dict, Any
from functools import wraps
import time

from loguru import logger
from llama_cpp import Llama

DEFAULT_MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf"

class TimeoutException(Exception):
    pass

def timeout(seconds=30):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise TimeoutException("Function execution timed out")
            
            # Set the timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the alarm
                signal.alarm(0)
            
            return result
        return wrapper
    return decorator

class LocalLLM:
    """
    A wrapper for the Llama-cpp-python model to handle loading and generation.
    Includes a 'dummy_mode' for development on machines without a powerful GPU.
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, dummy_mode: bool = False):
        self.model_path = Path(model_path)
        self.dummy_mode = dummy_mode
        # The _load_model method is called here, so the logic inside it is crucial
        self.llm = self._load_model()

    def _load_model(self) -> Llama | None:
        """Loads the GGUF model from the specified path, or returns None in dummy mode."""
        
        # Check for dummy mode FIRST, before we do anything else.
        if self.dummy_mode:
            logger.warning("LLM is running in Dummy Mode. No model will be loaded.")
            return None
            
        logger.info(f"Attempting to load LLM from: {self.model_path}")

        if not self.model_path.exists():
            logger.error(f"Model file not found at {self.model_path}")
            logger.error("Please download a GGUF model and place it in the 'models' directory.")
            logger.error(f"Update the DEFAULT_MODEL_PATH in core/llm.py to match the filename.")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            llm = Llama(
                model_path=str(self.model_path),
                n_ctx=2048,  # Reduced context for better performance
                n_gpu_layers=0,  # Force CPU-only for compatibility
                n_threads=4,  # Limit threads to prevent overloading
                verbose=False
            )
            logger.success("Local LLM loaded successfully into memory.")
            return llm
        except Exception as e:
            logger.error(f"Failed to load the LLM model: {e}")
            raise

    @timeout(30)  # 30 second timeout
    def generate_text(self, prompt: str, temperature: float = 0.2, max_tokens: int = 200) -> str:
        """Generate text response for a given prompt with timeout protection"""
        if self.dummy_mode:
            return "This is a simple text response."
        
        try:
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                stop=["END_PLAN", "END_EXECUTION", "END_CRITIQUE", "END_TRIAGE"]
            )
            return response['choices'][0]['message']['content'].strip()
        except TimeoutException:
            logger.warning("LLM generation timed out")
            return "I need more time to think about this. Please try again."
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            return "An error occurred while generating a response."

    def generate_trace_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generates a response from the LLM, asking for JSON output.
        If in dummy mode, returns a pre-defined dummy trace.
        """
        # The check is repeated here for safety, but the main logic is in _load_model
        if self.dummy_mode:
            logger.info("Dummy Mode: Returning a pre-defined test trace.")
            return self._get_dummy_trace()

        logger.info("Generating trace from real LLM...")
        
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a high-dimensional reasoning engine. Your sole output is a single, complete JSON object that conforms to the user's requested schema. Do not output any other text, greetings, or explanations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            
            json_string = response['choices'][0]['message']['content']
            logger.success("Successfully received JSON response from LLM.")
            return json.loads(json_string)

        except Exception as e:
            logger.error(f"Error during LLM generation or JSON parsing: {e}")
            raise

    def _get_dummy_trace(self) -> Dict[str, Any]:
        """Returns a hardcoded trace for testing without a real model."""
        import uuid
        from datetime import datetime, timezone
        return {
            "trace_id": f"dummy_trace_{uuid.uuid4()}",
            "query": "This is a dummy query from a laptop.",
            "seed_id": "eira-001",
            "summary": {
                "answer": "This is a dummy answer generated because the LLM is in dummy mode. The full application is working correctly.",
                "reasoning": "Dummy mode was activated, so a pre-defined response was returned to allow for testing the application flow without a GPU.",
                "next_action": "Switch to your main computer with a GPU and set `dummy_mode=False` in `interfaces/cli.py` to use the real model."
            },
            "attempts": [
                {
                    "id": 1,
                    "plan": ["Return the dummy data."],
                    "candidate": "This is a dummy answer.",
                    "scores": {"truth": 1.0, "helpfulness": 1.0, "clarity": 1.0, "ethics": 1.0},
                    "total": 1.0
                }
            ],
            "best": {
                "attempt_id": 1,
                "candidate": "This is a dummy answer.",
                "total": 1.0
            },
            "reflection": {
                "what_worked": "Dummy mode worked, allowing the application to run.",
                "what_failed": "No real reasoning occurred.",
                "next_adjustment": "Disable dummy mode when a real model is available."
            },
            "artifacts": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_info": {
                "name": "dummy_model",
                "runtime": "dummy_mode",
                "temp": 0.0
            }
        }