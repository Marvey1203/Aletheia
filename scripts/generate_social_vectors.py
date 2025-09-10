import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from loguru import logger

# --- Configuration ---
OUTPUT_DIR = Path("core_knowledge")
OUTPUT_FILE_PATH = OUTPUT_DIR / "social_vectors.npz"
MODEL_NAME = 'all-MiniLM-L6-v2' # Use the same model as our Atlas for consistency

# --- The Social Constitution ---
# This dictionary defines the core social concepts Aletheia will learn to perceive.
# The keys are the names of the concepts.
# The values are descriptive sentences that embody the meaning of each concept.
SOCIAL_CONCEPTS = {
    "FORMALITY": "This is a formal, serious, and professional request that requires a structured and respectful response.",
    "CASUALNESS": "This is a casual, friendly, and informal greeting or question that should be met with a relaxed and conversational tone.",
    "URGENCY": "This is an urgent, time-sensitive problem that requires an immediate, direct, and solution-focused answer.",
    "CURIOSITY": "This is an open-ended, curious, and exploratory question that invites a detailed, thoughtful, and insightful explanation.",
    "HUMOR": "This statement contains humor, wit, or sarcasm. The user is being playful and the response should acknowledge this lightheartedness.",
    "CORRECTION": "The user is providing a correction or pointing out a mistake. The response must be humble, acknowledge the correction, and integrate the new information.",
    "META_COGNITION": "The user is asking a question about my own nature, architecture, or thought process. The response should be self-referential and reflective."
}

def main():
    """
    Encodes the social constitution into a dictionary of named vectors
    and saves it for use by the Omega Social Acuity Core.
    """
    logger.info("--- Forging the Social Constitution Vectors ---")

    # 1. Initialize the embedding model
    logger.info(f"Loading the embedding model '{MODEL_NAME}'...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load the SentenceTransformer model. Is it installed? Error: {e}")
        return
    logger.success("Embedding model loaded.")

    # 2. Encode all concept descriptions
    logger.info(f"Encoding {len(SOCIAL_CONCEPTS)} social concepts into vectors...")
    
    concept_names = list(SOCIAL_CONCEPTS.keys())
    concept_descriptions = list(SOCIAL_CONCEPTS.values())
    
    concept_vectors = model.encode(concept_descriptions)
    
    # 3. Create the dictionary mapping names to vectors
    vector_dictionary = {name: vector for name, vector in zip(concept_names, concept_vectors)}
    
    logger.success("All social concepts have been encoded.")
    for name in vector_dictionary:
        print(f"  - Encoded '{name}' with shape: {vector_dictionary[name].shape}")

    # 4. Save the dictionary to a compressed .npz file
    logger.info(f"Saving the vector dictionary to '{OUTPUT_FILE_PATH}'...")
    OUTPUT_DIR.mkdir(exist_ok=True) # Ensure the output directory exists
    np.savez_compressed(OUTPUT_FILE_PATH, **vector_dictionary)
    logger.success("--- The Social Constitution has been forged and saved. ---")

if __name__ == "__main__":
    main()