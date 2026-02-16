"""Configuration module for Dynamic Agent Assembler."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class EmbeddingModelConfig(BaseModel):
    """Embedding model configuration."""
    name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True
    show_progress_bar: bool = False


class ChromaDBConfig(BaseModel):
    """ChromaDB configuration."""
    persist_directory: str = "./data/chromadb"
    collection_name: str = "agent_capabilities"
    distance_function: str = "cosine"


class VectorSearchConfig(BaseModel):
    """Vector search configuration."""
    embedding_model: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    default_top_k: int = 5
    min_similarity_score: float = 0.3
    enable_hybrid_search: bool = False


class TaskAnalysisConfig(BaseModel):
    """Task analysis configuration."""
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    enable_decomposition: bool = True
    max_subtasks: int = 10
    confidence_threshold: float = 0.7


class TeamAssemblyConfig(BaseModel):
    """Team assembly configuration."""
    default_team_type: str = "ephemeral"
    min_team_size: int = 1
    max_team_size: int = 10
    optimal_team_size: int = 3
    selection_strategy: str = "semantic_similarity"
    enable_role_assignment: bool = True
    reuse_threshold: float = 0.8
    cache_ttl: int = 3600


class PersistenceConfig(BaseModel):
    """Persistence configuration."""
    backend: str = "filesystem"
    data_directory: str = "./data"
    save_interval: int = 300
    persist_teams: bool = True
    team_retention_days: int = 7


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_tracing: bool = True


class Config(BaseModel):
    """Main configuration for Dynamic Agent Assembler."""
    vector_search: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    task_analysis: TaskAnalysisConfig = Field(default_factory=TaskAnalysisConfig)
    team_assembly: TeamAssemblyConfig = Field(default_factory=TeamAssemblyConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()
        
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        
        return cls(**data)

    @classmethod
    def from_default_locations(cls) -> "Config":
        """Load configuration from default locations."""
        # Check current directory
        default_paths = [
            Path("config.yaml"),
            Path("config.yml"),
            Path(__file__).parent.parent.parent / "config.yaml",
        ]
        
        for path in default_paths:
            if path.exists():
                return cls.from_yaml(path)
        
        return cls()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split(".")
        value = self.model_dump()
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
