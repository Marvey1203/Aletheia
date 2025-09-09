# test_model_loader.py
import sys
from pathlib import Path
from llama_cpp import Llama

def test_load(model_path_str: str):
    """A simple script to test if a GGUF model can be loaded."""
    print(f"--- Attempting to load model: {model_path_str} ---")
    model_path = Path(model_path_str)

    if not model_path.exists():
        print(f"❌ ERROR: Model file not found at the specified path.")
        return

    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_gpu_layers=-1, # Load all possible layers to GPU
            verbose=True
        )
        print("\n✅ SUCCESS: Model loaded successfully into memory.")
    except Exception as e:
        print(f"\n❌ FAILED: An error occurred while loading the model.")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model_loader.py <path_to_model.gguf>")
        sys.exit(1)
    
    test_model_path = sys.argv[1]
    test_load(test_model_path)
