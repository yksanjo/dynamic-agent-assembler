"""Main Dynamic Agent Assembler that integrates all components."""

from pathlib import Path
from typing import Any, Optional

from dynamic_agent_assembler.agent_assembler import AgentAssembler, TeamMember
from dynamic_agent_assembler.capability_registry import AgentCapability, CapabilityCategory, CapabilityRegistry
from dynamic_agent_assembler.config import Config
from dynamic_agent_assembler.task_analyzer import Task, TaskAnalyzer
from dynamic_agent_assembler.team_manager import AgentTeam, TeamManager, TeamType
from dynamic_agent_assembler.vector_search import VectorSearchEngine


class DynamicAgentAssembler:
    """
    Main class for dynamic agent team construction.
    
    Integrates:
    - Vector Search Engine for capability matching
    - Capability Registry for agent management
    - Task Analyzer for task decomposition
    - Agent Assembler for team construction
    - Team Manager for team lifecycle management
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_default_locations()
        
        # Initialize components
        self.vector_engine: Optional[VectorSearchEngine] = None
        self.capability_registry: Optional[CapabilityRegistry] = None
        self.task_analyzer: Optional[TaskAnalyzer] = None
        self.agent_assembler: Optional[AgentAssembler] = None
        self.team_manager: Optional[TeamManager] = None
        
        # Track initialization state
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all components."""
        # Initialize vector search engine
        vs_config = self.config.vector_search
        self.vector_engine = VectorSearchEngine(
            model_name=vs_config.embedding_model.name,
            device=vs_config.embedding_model.device,
            persist_directory=vs_config.chromadb.persist_directory,
            collection_name=vs_config.chromadb.collection_name,
            distance_function=vs_config.chromadb.distance_function,
        )
        self.vector_engine.initialize()
        
        # Initialize capability registry
        self.capability_registry = CapabilityRegistry(vector_engine=self.vector_engine)
        
        # Initialize task analyzer
        ta_config = self.config.task_analysis
        self.task_analyzer = TaskAnalyzer(
            provider=ta_config.provider,
            model=ta_config.model,
            temperature=ta_config.temperature,
            max_tokens=ta_config.max_tokens,
            enable_decomposition=ta_config.enable_decomposition,
            max_subtasks=ta_config.max_subtasks,
            confidence_threshold=ta_config.confidence_threshold,
        )
        
        # Initialize agent assembler
        ta_config = self.config.team_assembly
        self.agent_assembler = AgentAssembler(
            vector_engine=self.vector_engine,
            capability_registry=self.capability_registry,
            min_team_size=ta_config.min_team_size,
            max_team_size=ta_config.max_team_size,
            optimal_team_size=ta_config.optimal_team_size,
            selection_strategy=ta_config.selection_strategy,
            enable_role_assignment=ta_config.enable_role_assignment,
            min_similarity_score=self.config.vector_search.search.min_similarity_score,
        )
        
        # Initialize team manager
        self.team_manager = TeamManager(
            agent_assembler=self.agent_assembler,
            team_type=ta_config.default_team_type,
            min_team_size=ta_config.min_team_size,
            max_team_size=ta_config.max_team_size,
            reuse_threshold=ta_config.reuse_threshold,
            cache_ttl=ta_config.cache_ttl,
        )
        
        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if the assembler is initialized."""
        return self._initialized

    def ensure_initialized(self) -> None:
        """Ensure the assembler is initialized."""
        if not self._initialized:
            self.initialize()

    # ============ Capability Registry Methods ============

    def register_agent(
        self,
        agent_id: str,
        agent_name: str,
        description: str,
        capabilities: list[str],
        category: str = "specialized",
        keywords: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentCapability:
        """Register a new agent with its capabilities."""
        self.ensure_initialized()
        
        capability = AgentCapability(
            agent_id=agent_id,
            agent_name=agent_name,
            description=description,
            capabilities=capabilities,
            category=CapabilityCategory(category),
            keywords=keywords or [],
            metadata=metadata or {},
        )
        
        self.capability_registry.register(capability)
        return capability

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        self.ensure_initialized()
        
        capability = self.capability_registry.get_by_agent_id(agent_id)
        if capability:
            return self.capability_registry.unregister(capability.id)
        return False

    def list_agents(self) -> list[AgentCapability]:
        """List all registered agents."""
        self.ensure_initialized()
        return self.capability_registry.list_active()

    # ============ Task Analysis Methods ============

    def analyze_task(
        self,
        description: str,
        context: Optional[str] = None,
        priority: str = "medium",
    ) -> Task:
        """Analyze a task and decompose it into subtasks."""
        self.ensure_initialized()
        
        task = Task(
            description=description,
            context=context,
            priority=priority,
        )
        
        return self.task_analyzer.analyze(task)

    # ============ Team Construction Methods ============

    def build_team(
        self,
        task: Task,
        team_type: str = "ephemeral",
        team_name: Optional[str] = None,
    ) -> AgentTeam:
        """Build a team for a task."""
        self.ensure_initialized()
        
        # Set team type
        self.team_manager.set_team_type(team_type)
        
        # Create team
        return self.team_manager.create_team(task, team_name=team_name)

    def build_team_from_description(
        self,
        description: str,
        context: Optional[str] = None,
        team_type: str = "ephemeral",
        team_name: Optional[str] = None,
    ) -> AgentTeam:
        """Build a team from a task description (convenience method)."""
        # Analyze task first
        task = self.analyze_task(description, context)
        
        # Build team
        return self.build_team(task, team_type=team_type, team_name=team_name)

    # ============ Team Management Methods ============

    def get_team(self, team_id: str) -> Optional[AgentTeam]:
        """Get a team by ID."""
        from uuid import UUID
        try:
            return self.team_manager.get_team(UUID(team_id))
        except ValueError:
            return None

    def list_teams(self, status: Optional[str] = None) -> list[AgentTeam]:
        """List all teams."""
        from dynamic_agent_assembler.team_manager import TeamStatus
        status_enum = TeamStatus(status) if status else None
        return self.team_manager.list_teams(status=status_enum)

    def dissolve_team(self, team_id: str, reason: Optional[str] = None) -> bool:
        """Dissolve a team."""
        from uuid import UUID
        try:
            return self.team_manager.dissolve_team(UUID(team_id), reason=reason)
        except ValueError:
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get assembler statistics."""
        self.ensure_initialized()
        
        return {
            "registered_agents": self.capability_registry.count(),
            "team_stats": self.team_manager.get_team_stats(),
            "config": {
                "team_type": self.config.team_assembly.default_team_type,
                "embedding_model": self.config.vector_search.embedding_model.name,
            },
        }

    # ============ Search Methods ============

    def search_agents(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[AgentCapability]:
        """Search for agents matching a query."""
        self.ensure_initialized()
        
        if self.vector_engine and self.vector_engine.is_initialized():
            results = self.vector_engine.search(
                query=query,
                top_k=top_k,
                min_similarity=self.config.vector_search.search.min_similarity_score,
            )
            return [r.capability for r in results]
        else:
            return self.capability_registry.search_by_text(query, limit=top_k)

    # ============ Cleanup Methods ============

    def shutdown(self) -> None:
        """Shutdown the assembler and cleanup resources."""
        if self.team_manager:
            self.team_manager.shutdown()
        
        self._initialized = False

    def clear_all(self) -> None:
        """Clear all registered agents and teams."""
        self.ensure_initialized()
        
        self.capability_registry.clear()
        self.team_manager.shutdown()
