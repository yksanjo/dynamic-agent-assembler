"""Agent Assembler module for dynamic team construction."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from dynamic_agent_assembler.capability_registry import AgentCapability
from dynamic_agent_assembler.task_analyzer import Task, SubTask
from dynamic_agent_assembler.vector_search import SearchResult


class AgentRole(str, Enum):
    """Roles that agents can play in a team."""
    LEADER = "leader"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"


class TeamStrategy(str, Enum):
    """Team assembly strategies."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    WEIGHTED = "weighted"
    ENSEMBLE = "ensemble"
    GREEDY = "greedy"


@dataclass
class TeamMember:
    """Represents a team member with assigned role."""
    capability: AgentCapability
    role: AgentRole
    assigned_subtasks: list[UUID] = None
    score: float = 0.0

    def __post_init__(self):
        if self.assigned_subtasks is None:
            self.assigned_subtasks = []


class AgentAssembler:
    """Assembles agent teams based on task requirements and capability matching."""

    def __init__(
        self,
        vector_engine: Optional[Any] = None,
        capability_registry: Optional[Any] = None,
        min_team_size: int = 1,
        max_team_size: int = 10,
        optimal_team_size: int = 3,
        selection_strategy: str = "semantic_similarity",
        enable_role_assignment: bool = True,
        min_similarity_score: float = 0.3,
    ):
        self.vector_engine = vector_engine
        self.capability_registry = capability_registry
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size
        self.optimal_team_size = optimal_team_size
        self.selection_strategy = selection_strategy
        self.enable_role_assignment = enable_role_assignment
        self.min_similarity_score = min_similarity_score

    def assemble_team(
        self,
        task: Task,
        top_k: Optional[int] = None,
    ) -> list[TeamMember]:
        """Assemble a team for a given task."""
        if top_k is None:
            top_k = self.optimal_team_size * 2  # Get more candidates for selection

        # Search for matching capabilities
        search_results = self._search_capabilities(task, top_k)
        
        if not search_results:
            return []

        # Select optimal team members
        team_members = self._select_team_members(task, search_results)
        
        # Assign roles if enabled
        if self.enable_role_assignment:
            team_members = self._assign_roles(team_members)
        
        return team_members

    def _search_capabilities(
        self,
        task: Task,
        top_k: int,
    ) -> list[SearchResult]:
        """Search for capabilities matching task requirements."""
        # Build search query from task requirements
        query = self._build_search_query(task)
        
        if self.vector_engine and self.vector_engine.is_initialized():
            # Use vector search
            return self.vector_engine.search(
                query=query,
                top_k=top_k,
                min_similarity=self.min_similarity_score,
            )
        elif self.capability_registry:
            # Fallback to text search
            capabilities = self.capability_registry.search_by_text(
                query=query,
                limit=top_k,
            )
            # Convert to SearchResult format
            return [
                SearchResult(
                    capability=cap,
                    score=1.0,
                    distance=0.0,
                )
                for cap in capabilities
            ]
        
        return []

    def _build_search_query(self, task: Task) -> str:
        """Build search query from task and subtask requirements."""
        parts = [task.description]
        
        # Add required capabilities
        if task.required_capabilities:
            parts.extend(task.required_capabilities)
        
        # Add subtask requirements
        for subtask in task.subtasks:
            parts.append(subtask.description)
            if subtask.required_capabilities:
                parts.extend(subtask.required_capabilities)
        
        return " ".join(parts)

    def _select_team_members(
        self,
        task: Task,
        search_results: list[SearchResult],
    ) -> list[TeamMember]:
        """Select optimal team members from search results."""
        if not search_results:
            return []

        # Apply selection strategy
        if self.selection_strategy == TeamStrategy.SEMANTIC_SIMILARITY.value:
            return self._select_by_similarity(search_results)
        elif self.selection_strategy == TeamStrategy.WEIGHTED.value:
            return self._select_weighted(task, search_results)
        elif self.selection_strategy == TeamStrategy.GREEDY.value:
            return self._select_greedy(task, search_results)
        else:
            return self._select_by_similarity(search_results)

    def _select_by_similarity(
        self,
        search_results: list[SearchResult],
    ) -> list[TeamMember]:
        """Select team members based on similarity scores."""
        # Filter by minimum score
        valid_results = [
            r for r in search_results
            if r.score >= self.min_similarity_score
        ]
        
        # Select top candidates
        team_members = []
        for result in valid_results[:self.max_team_size]:
            team_members.append(TeamMember(
                capability=result.capability,
                role=AgentRole.SPECIALIST,
                score=result.score,
            ))
        
        # Ensure minimum team size
        if len(team_members) < self.min_team_size:
            # Add more from results if available
            for result in search_results[len(team_members):]:
                if result.capability.agent_id not in [m.capability.agent_id for m in team_members]:
                    team_members.append(TeamMember(
                        capability=result.capability,
                        role=AgentRole.SPECIALIST,
                        score=result.score,
                    ))
                    if len(team_members) >= self.min_team_size:
                        break
        
        return team_members[:self.max_team_size]

    def _select_weighted(
        self,
        task: Task,
        search_results: list[SearchResult],
    ) -> list[TeamMember]:
        """Select team members using weighted scoring based on task requirements."""
        # Calculate weighted scores
        weighted_results = []
        
        for result in search_results:
            weight = self._calculate_weight(task, result)
            weighted_score = result.score * weight
            weighted_results.append((result, weighted_score))
        
        # Sort by weighted score
        weighted_results.sort(key=lambda x: x[1], reverse=True)
        
        team_members = []
        selected_ids = set()
        
        for result, score in weighted_results:
            if len(team_members) >= self.max_team_size:
                break
            
            agent_id = result.capability.agent_id
            if agent_id not in selected_ids:
                team_members.append(TeamMember(
                    capability=result.capability,
                    role=AgentRole.SPECIALIST,
                    score=score,
                ))
                selected_ids.add(agent_id)
        
        return team_members

    def _calculate_weight(self, task: Task, result: SearchResult) -> float:
        """Calculate weight for a capability based on task requirements."""
        capability = result.capability
        weight = 1.0
        
        # Boost if capability matches required capabilities
        required_caps = set(task.required_capabilities)
        agent_caps = set(capability.capabilities)
        
        if required_caps & agent_caps:
            weight *= 1.5
        
        # Boost based on category relevance
        if task.subtasks:
            # Check if category matches
            weight *= 1.0
        
        return weight

    def _select_greedy(
        self,
        task: Task,
        search_results: list[SearchResult],
    ) -> list[TeamMember]:
        """Greedily select team members to cover all requirements."""
        team_members = []
        covered_capabilities = set()
        
        # Get all required capabilities
        all_required = set(task.required_capabilities)
        for subtask in task.subtasks:
            all_required.update(subtask.required_capabilities)
        
        # Greedy selection
        for result in search_results:
            if len(team_members) >= self.max_team_size:
                break
            
            agent_caps = set(result.capability.capabilities)
            new_coverage = all_required - covered_capabilities
            
            # Check if this agent adds new coverage
            if agent_caps & new_coverage:
                team_members.append(TeamMember(
                    capability=result.capability,
                    role=AgentRole.SPECIALIST,
                    score=result.score,
                ))
                covered_capabilities.update(agent_caps)
        
        # Ensure minimum team size
        if len(team_members) < self.min_team_size:
            for result in search_results:
                if len(team_members) >= self.min_team_size:
                    break
                if result.capability.agent_id not in [m.capability.agent_id for m in team_members]:
                    team_members.append(TeamMember(
                        capability=result.capability,
                        role=AgentRole.SPECIALIST,
                        score=result.score,
                    ))
        
        return team_members

    def _assign_roles(self, team_members: list[TeamMember]) -> list[TeamMember]:
        """Assign roles to team members."""
        if not team_members:
            return team_members
        
        # Sort by score (highest first)
        sorted_members = sorted(team_members, key=lambda m: m.score, reverse=True)
        
        # Assign roles
        if len(sorted_members) == 1:
            sorted_members[0].role = AgentRole.LEADER
        elif len(sorted_members) == 2:
            sorted_members[0].role = AgentRole.LEADER
            sorted_members[1].role = AgentRole.EXECUTOR
        else:
            # First is leader
            sorted_members[0].role = AgentRole.LEADER
            # Last is reviewer if we have enough members
            if len(sorted_members) > 2:
                sorted_members[-1].role = AgentRole.REVIEWER
            # Middle members are specialists/executors
            for i in range(1, len(sorted_members) - 1):
                if sorted_members[i].role == AgentRole.SPECIALIST:
                    sorted_members[i].role = AgentRole.COORDINATOR if i == 1 else AgentRole.SPECIALIST
        
        return sorted_members

    def reassign_subtasks(
        self,
        team_members: list[TeamMember],
        task: Task,
    ) -> list[TeamMember]:
        """Reassign subtasks to team members based on capabilities."""
        for subtask in task.subtasks:
            # Find best matching agent
            best_member = None
            best_score = 0.0
            
            for member in team_members:
                agent_caps = set(member.capability.capabilities)
                required = set(subtask.required_capabilities)
                
                # Calculate match score
                if agent_caps & required:
                    overlap = len(agent_caps & required) / len(required)
                    if overlap > best_score:
                        best_score = overlap
                        best_member = member
            
            if best_member:
                best_member.assigned_subtasks.append(subtask.id)
        
        return team_members
