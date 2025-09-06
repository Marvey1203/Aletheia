# core/identity.py
# This file defines the IdentityCore, which is responsible for loading and
# managing the AI's constitutional seed file.

import json
from pathlib import Path
from typing import Dict, List, Any

from loguru import logger

class IdentityCore:
    """
    Manages the loading and accessibility of the AI's core identity from a seed file.
    """
    def __init__(self, seed_id: str = "eira-001"):
        self.seed_id = seed_id
        self.seed_data: Dict[str, Any] = self._load_seed()
        
        # Provide easy, direct access to core components
        self.identity_data = self.seed_data.get("identity_core", {})
        self.weights = self.identity_data.get("weights", {})
        self.principles = self.identity_data.get("principles", [])
        self.drives = self.identity_data.get("intrinsic_drives", [])
        logger.info(f"Identity Core for '{self.seed_id}' initialized.")

    def _load_seed(self) -> Dict[str, Any]:
        """
        Loads the specified seed JSON file from the seeds directory.
        """
        # We assume the script is run from the root of the project.
        seed_path = Path(f"seeds/{self.seed_id}_seed.json")
        logger.info(f"Loading identity core from: {seed_path}")

        if not seed_path.exists():
            logger.error(f"Seed file not found at {seed_path}")
            raise FileNotFoundError(f"Could not find seed file: {seed_path}. Make sure you are running from the project root.")

        with open(seed_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                logger.success("Identity Core loaded successfully.")
                return data
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {seed_path}: {e}")
                raise
    
    def get_weights(self) -> Dict[str, float]:
        """Returns the scoring weights from the identity core."""
        return self.weights

    def get_principles(self) -> List[str]:
        """Returns the list of core principles."""
        return self.principles

# Example of how to use it (for standalone testing of this module)
if __name__ == "__main__":
    try:
        # This assumes you run `python -m core.identity` from the `aletheia` root directory
        identity = IdentityCore()
        print("\n--- Identity Core Test ---")
        print("Successfully loaded Identity Core.")
        print("Weights:", identity.get_weights())
        print("First Principle:", identity.get_principles()[0])
        print("--------------------------")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")