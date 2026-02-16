"""Vector Search Engine module with ChromaDB integration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from dynamic_agent_assembler.capability_registry import AgentCapability


@dataclass
class SearchResult:
    """Represents a search result."""
    capability: AgentCapability
    score: float
    distance: float


class VectorSearchEngine:
    """Vector search engine using ChromaDB and sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        persist_directory: str = "./data/chromadb",
        collection_name: str = "agent_capabilities",
        distance_function: str = "cosine",
    ):
        self.model_name = model_name
        self.device = device
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.distance_function = distance_function
        
        self._model = None
        self._client = None
        self._collection = None
        self._capability_cache: dict[str, AgentCapability] = {}

    def initialize(self) -> None:
        """Initialize the vector search engine."""
        # Import here to make dependencies optional
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
        except ImportError as e:
            raise ImportError(
                "Required packages not installed. Run: pip install dynamic-agent-assembler"
            ) from e
        
        # Load embedding model
        self._model = SentenceTransformer(self.model_name, device=self.device)
        
        # Initialize ChromaDB client
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_function}
        )

    def is_initialized(self) -> bool:
        """Check if the engine is initialized."""
        return self._model is not None and self._collection is not None

    def _ensure_initialized(self) -> None:
        """Ensure the engine is initialized."""
        if not self.is_initialized():
            self.initialize()

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        self._ensure_initialized()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to the vector store."""
        self._ensure_initialized()
        
        search_text = capability.to_search_text()
        embedding = self._get_embedding(search_text)
        
        self._collection.upsert(
            ids=[str(capability.id)],
            embeddings=[embedding],
            documents=[search_text],
            metadatas=[{
                "agent_id": capability.agent_id,
                "agent_name": capability.agent_name,
                "category": capability.category.value,
                "is_active": capability.is_active,
            }]
        )
        
        self._capability_cache[str(capability.id)] = capability

    def remove_capability(self, capability_id: str) -> None:
        """Remove a capability from the vector store."""
        self._ensure_initialized()
        
        self._collection.delete(ids=[capability_id])
        self._capability_cache.pop(capability_id, None)

    def update_capability(self, capability: AgentCapability) -> None:
        """Update a capability in the vector store."""
        self.add_capability(capability)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        category_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search for capabilities matching a query."""
        self._ensure_initialized()
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Build where clause for filtering
        where = None
        if category_filter:
            where = {"category": category_filter}
        
        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances", "documents"]
        )
        
        # Process results
        search_results = []
        if results and results["ids"] and len(results["ids"]) > 0:
            for i, cap_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                
                # Convert distance to similarity score (for cosine)
                score = 1 - distance
                
                if score < min_similarity:
                    continue
                
                # Get capability from cache or recreate
                capability = self._capability_cache.get(cap_id)
                if not capability:
                    # Try to get metadata
                    metadata = results["metadatas"][0][i]
                    capability = AgentCapability(
                        agent_id=metadata.get("agent_id", ""),
                        agent_name=metadata.get("agent_name", ""),
                        description="",
                    )
                
                search_results.append(SearchResult(
                    capability=capability,
                    score=score,
                    distance=distance,
                ))
        
        return search_results

    def search_by_capabilities(
        self,
        required_capabilities: list[str],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search for capabilities matching a list of required capabilities."""
        # Combine capabilities into a single query
        query = " ".join(required_capabilities)
        return self.search(query, top_k)

    def get_all_capabilities(self) -> list[AgentCapability]:
        """Get all capabilities in the vector store."""
        self._ensure_initialized()
        
        results = self._collection.get()
        capabilities = []
        
        if results and results["ids"]:
            for i, cap_id in enumerate(results["ids"][0]):
                capability = self._capability_cache.get(cap_id)
                if capability:
                    capabilities.append(capability)
        
        return capabilities

    def clear(self) -> None:
        """Clear all capabilities from the vector store."""
        self._ensure_initialized()
        
        self._collection.delete(where={})
        self._capability_cache.clear()

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        self._ensure_initialized()
        
        if self._model is None:
            return 384  # Default for all-MiniLM-L6-v2
        
        return self._model.get_sentence_embedding_dimension()
