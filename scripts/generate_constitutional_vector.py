import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from loguru import logger

# --- Configuration ---
SEED_FILE_PATH = Path("seeds/eira-001_seed.json")
OUTPUT_DIR = Path("core_knowledge")
OUTPUT_FILE_PATH = OUTPUT_DIR / "constitutional_vector.npy"
MODEL_NAME = 'all-MiniLM-L6-v2' # Use the same model as our Atlas for consistency

def main():
    """
    Reads Aletheia's principles from the seed file, encodes them into vectors,
    and saves the averaged "Constitutional Vector" for use by the Omega Core.
    """
    logger.info("--- Forging the Constitutional Vector ---")

    # 1. Load the seed file
    logger.info(f"Loading identity principles from '{SEED_FILE_PATH}'...")
    if not SEED_FILE_PATH.exists():
        logger.error(f"Seed file not found! Make sure '{SEED_FILE_PATH}' exists.")
        return
    
    with open(SEED_FILE_PATH, "r") as f:
        seed_data = json.load(f)
    
    principles = seed_data.get("identity_core", {}).get("principles", [])
    if not principles:
        logger.error("No principles found in the seed file.")
        return
        
    logger.info(f"Found {len(principles)} principles to encode.")
    for i, p in enumerate(principles):
        print(f"  {i+1}. {p}")

    # 2. Initialize the embedding model
    logger.info(f"Loading the embedding model '{MODEL_NAME}'...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load the SentenceTransformer model. Is it installed? Error: {e}")
        return
    logger.success("Embedding model loaded.")

    # 3. Encode principles into vectors
    logger.info("Encoding principles into high-dimensional vectors...")
    principle_vectors = model.encode(principles)
    logger.success("All principles have been encoded.")

    # 4. Calculate the unified Constitutional Vector
    logger.info("Calculating the unified vector (the geometric mean)...")
    constitutional_vector = np.mean(principle_vectors, axis=0)
    logger.success(f"Constitutional Vector created with shape: {constitutional_vector.shape}")

    # 5. Save the vector to a file
    logger.info(f"Saving the vector to '{OUTPUT_FILE_PATH}'...")
    OUTPUT_DIR.mkdir(exist_ok=True) # Ensure the output directory exists
    np.save(OUTPUT_FILE_PATH, constitutional_vector)
    logger.success("--- The Constitutional Vector has been forged and saved. ---")

if __name__ == "__main__":
    main()