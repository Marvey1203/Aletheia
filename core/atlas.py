# core/atlas.py

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from pathlib import Path

from loguru import logger
from .schemas import Trace
from .llm import ModelManager

class ConceptualAtlas:
    """
    Manages the vector database for Aletheia's long-term memory.
    This is the implementation of the "Conceptual Atlas" from the Codex.
    """
    def __init__(self, model_manager: ModelManager, db_path: str = "memory_galaxy/atlas_db"):
        logger.info("Initializing Conceptual Atlas...")
        self.model_manager = model_manager
        
        # Initialize the ChromaDB client
        # This will create the directory if it doesn't exist.
        self.client = chromadb.PersistentClient(path=db_path)

        # Create a custom embedding function that uses our ModelManager
        # We need to properly initialize our custom class with the model_manager
        class AletheiaEmbeddingFunction(embedding_functions.EmbeddingFunction):
            # The __init__ method for our custom class
            def __init__(self, model_manager_instance: ModelManager):
                self.model_manager = model_manager_instance

            def __call__(self, input: List[str]) -> List[List[float]]:
                logger.info(f"Atlas: Embedding batch of {len(input)} documents...")
                # Now `self.model_manager` will exist
                embeddings = self.model_manager.create_embedding(input)
                logger.success("Embedding batch complete.")
                return embeddings
        
        # Instantiate our custom embedding function, passing self.model_manager to it
        aletheia_embedder = AletheiaEmbeddingFunction(self.model_manager)

        # Get or create the main collection for traces
        self.trace_collection = self.client.get_or_create_collection(
            name="traces",
            embedding_function=aletheia_embedder
        )
        logger.success(f"Conceptual Atlas is ready. Collection 'traces' has {self.trace_collection.count()} items.")

    def _prepare_trace_for_embedding(self, trace: Trace) -> tuple[str, dict]:
        """
        Prepares a trace for storage by creating the document to be embedded
        and the associated metadata.
        """
        # The text we embed should be the most meaningful part of the trace
        document_text = f"User Query: {trace.query}\nAletheia's Answer: {trace.summary.answer}"
        
        # The metadata allows us to store structured data alongside the vector
        metadata = {
            "query": trace.query,
            "answer": trace.summary.answer,
            "reasoning": trace.summary.reasoning,
            "timestamp": trace.timestamp,
            "best_score": trace.best.total
        }
        
        return document_text, metadata

    def add_trace(self, trace: Trace):
        """
        Embeds a trace and adds it to the vector database.
        """
        document, metadata = self._prepare_trace_for_embedding(trace)
        
        # The trace_id is the unique identifier for our document
        trace_id = trace.trace_id
        
        try:
            # Upsert will add the document if it's new, or update it if the ID already exists.
            self.trace_collection.upsert(
                ids=[trace_id],
                documents=[document],
                metadatas=[metadata]
            )
            logger.success(f"Successfully added/updated trace '{trace_id}' in the Conceptual Atlas.")
        except Exception as e:
            logger.error(f"Failed to add trace '{trace_id}' to Atlas: {e}")

    def query_atlas(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a semantic search on the Atlas to find the most relevant traces.
        """
        if self.trace_collection.count() == 0:
            logger.warning("Atlas query attempted on an empty collection.")
            return []

        logger.info(f"Querying Atlas for '{query_text[:50]}...' with {n_results} results.")
        try:
            results = self.trace_collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # The result object is complex, so we'll simplify it for the context injector.
            # We are interested in the metadata of the retrieved documents.
            retrieved_metadatas = results.get('metadatas', [[]])[0]
            logger.success(f"Atlas query returned {len(retrieved_metadatas)} relevant memories.")
            return retrieved_metadatas
            
        except Exception as e:
            logger.error(f"Atlas query failed: {e}")
            return []