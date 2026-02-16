"""Capability Registry module for managing agent capabilities."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CapabilityCategory(str, Enum):
    """Categories of agent capabilities."""
    REASONING = "reasoning"
    CREATION = "creation"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    COORDINATION = "coordination"
    SPECIALIZED = "specialized"


class AgentCapability(BaseModel):
    """Represents an agent's capability."""
    id: UUID = Field(default_factory=uuid4)
    agent_id: str
    agent_name: str
    description: str
    capabilities: list[str] = Field(default_factory=list)
    category: CapabilityCategory = CapabilityCategory.SPECIALIZED
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    is_active: bool = True

    def to_search_text(self) -> str:
        """Convert capability to searchable text."""
        parts = [
            self.agent_name,
            self.description,
            *self.capabilities,
            *self.keywords,
        ]
        return " ".join(parts)

    def update(self, **kwargs) -> None:
        """Update capability fields."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()
        self.version += 1


class CapabilityRegistry:
    """Registry for managing agent capabilities with vector search."""

    def __init__(self, vector_engine: Optional[Any] = None):
        self._capabilities: dict[UUID, AgentCapability] = {}
        self._agent_index: dict[str, UUID] = {}
        self._vector_engine = vector_engine

    def register(self, capability: AgentCapability) -> UUID:
        """Register a new agent capability."""
        self._capabilities[capability.id] = capability
        self._agent_index[capability.agent_id] = capability.id
        
        if self._vector_engine:
            self._vector_engine.add_capability(capability)
        
        return capability.id

    def unregister(self, capability_id: UUID) -> bool:
        """Unregister an agent capability."""
        if capability_id not in self._capabilities:
            return False
        
        capability = self._capabilities[capability_id]
        del self._capabilities[capability_id]
        del self._agent_index[capability.agent_id]
        
        if self._vector_engine:
            self._vector_engine.remove_capability(str(capability_id))
        
        return True

    def get(self, capability_id: UUID) -> Optional[AgentCapability]:
        """Get a capability by ID."""
        return self._capabilities.get(capability_id)

    def get_by_agent_id(self, agent_id: str) -> Optional[AgentCapability]:
        """Get a capability by agent ID."""
        cid = self._agent_index.get(agent_id)
        return self._capabilities.get(cid) if cid else None

    def update(self, capability: AgentCapability) -> bool:
        """Update an existing capability."""
        if capability.id not in self._capabilities:
            return False
        
        self._capabilities[capability.id] = capability
        
        if self._vector_engine:
            self._vector_engine.update_capability(capability)
        
        return True

    def list_all(self) -> list[AgentCapability]:
        """List all registered capabilities."""
        return list(self._capabilities.values())

    def list_by_category(self, category: CapabilityCategory) -> list[AgentCapability]:
        """List capabilities by category."""
        return [c for c in self._capabilities.values() if c.category == category]

    def list_active(self) -> list[AgentCapability]:
        """List all active capabilities."""
        return [c for c in self._capabilities.values() if c.is_active]

    def search_by_text(self, query: str, limit: int = 5) -> list[AgentCapability]:
        """Search capabilities by text (fallback when no vector engine)."""
        query_lower = query.lower()
        results = []
        
        for capability in self._capabilities.values():
            if not capability.is_active:
                continue
            
            search_text = capability.to_search_text().lower()
            if query_lower in search_text:
                results.append(capability)
            
            # Check keyword matches
            for keyword in capability.keywords:
                if query_lower in keyword.lower():
                    results.append(capability)
                    break
        
        return results[:limit]

    def count(self) -> int:
        """Get total number of registered capabilities."""
        return len(self._capabilities)

    def clear(self) -> None:
        """Clear all capabilities."""
        self._capabilities.clear()
        self._agent_index.clear()
        
        if self._vector_engine:
            self._vector_engine.clear()
