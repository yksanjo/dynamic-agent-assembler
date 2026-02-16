"""Team Manager module for managing agent teams (Ephemeral, Persistent, Hybrid)."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from dynamic_agent_assembler.agent_assembler import TeamMember
from dynamic_agent_assembler.task_analyzer import Task


class TeamType(str, Enum):
    """Type of team: ephemeral, persistent, or hybrid."""
    EPHEMERAL = "ephemeral"
    PERSISTENT = "persistent"
    HYBRID = "hybrid"


class TeamStatus(str, Enum):
    """Status of an agent team."""
    FORMING = "forming"
    ACTIVE = "active"
    IDLE = "idle"
    DISSOLVING = "dissolving"
    DISSOLVED = "dissolved"


class AgentTeam(BaseModel):
    """Represents an assembled agent team."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    team_type: TeamType = TeamType.EPHEMERAL
    status: TeamStatus = TeamStatus.FORMING
    members: list[TeamMember] = Field(default_factory=list)
    tasks: list[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_member(self, member: TeamMember) -> None:
        """Add a member to the team."""
        self.members.append(member)
        self.last_active_at = datetime.utcnow()

    def remove_member(self, agent_id: str) -> bool:
        """Remove a member from the team."""
        for i, member in enumerate(self.members):
            if member.capability.agent_id == agent_id:
                self.members.pop(i)
                self.last_active_at = datetime.utcnow()
                return True
        return False

    def get_leader(self) -> Optional[TeamMember]:
        """Get the team leader."""
        for member in self.members:
            if member.role.value == "leader":
                return member
        return self.members[0] if self.members else None

    def get_member_count(self) -> int:
        """Get the number of team members."""
        return len(self.members)

    def is_active(self) -> bool:
        """Check if the team is active."""
        return self.status == TeamStatus.ACTIVE


class TeamCache:
    """Cache for persistent teams."""

    def __init__(self, ttl: int = 3600, reuse_threshold: float = 0.8):
        self.ttl = ttl
        self.reuse_threshold = reuse_threshold
        self._cache: dict[str, tuple[AgentTeam, datetime]] = {}
        self._access_count: dict[str, int] = {}

    def get(self, task_signature: str) -> Optional[AgentTeam]:
        """Get a cached team for a task signature."""
        if task_signature not in self._cache:
            return None
        
        team, cached_at = self._cache[task_signature]
        
        # Check if cache is still valid
        if datetime.utcnow() - cached_at > timedelta(seconds=self.ttl):
            del self._cache[task_signature]
            return None
        
        # Update access count
        self._access_count[task_signature] = self._access_count.get(task_signature, 0) + 1
        
        return team

    def put(self, task_signature: str, team: AgentTeam) -> None:
        """Cache a team for a task signature."""
        self._cache[task_signature] = (team, datetime.utcnow())
        self._access_count[task_signature] = 1

    def invalidate(self, task_signature: str) -> None:
        """Invalidate a cached team."""
        self._cache.pop(task_signature, None)
        self._access_count.pop(task_signature, None)

    def clear(self) -> None:
        """Clear all cached teams."""
        self._cache.clear()
        self._access_count.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "total_accesses": sum(self._access_count.values()),
            "avg_accesses": sum(self._access_count.values()) / len(self._access_count) if self._access_count else 0,
        }


class TeamManager:
    """Manages agent teams with support for ephemeral, persistent, and hybrid modes."""

    def __init__(
        self,
        agent_assembler: Optional[Any] = None,
        team_type: str = "ephemeral",
        min_team_size: int = 1,
        max_team_size: int = 10,
        reuse_threshold: float = 0.8,
        cache_ttl: int = 3600,
    ):
        self.agent_assembler = agent_assembler
        self.team_type = TeamType(team_type)
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size
        self.reuse_threshold = reuse_threshold
        self.cache_ttl = cache_ttl
        
        # Active teams
        self._active_teams: dict[UUID, AgentTeam] = {}
        
        # Team cache for persistent/hybrid modes
        self._team_cache = TeamCache(ttl=cache_ttl, reuse_threshold=reuse_threshold)

    def create_team(
        self,
        task: Task,
        team_name: Optional[str] = None,
        force_new: bool = False,
    ) -> AgentTeam:
        """Create a team for a task."""
        team_id = uuid4()
        team_name = team_name or f"team-{team_id.hex[:8]}"

        # Check cache for persistent/hybrid modes
        if not force_new and self.team_type in (TeamType.PERSISTENT, TeamType.HYBRID):
            task_sig = self._get_task_signature(task)
            cached_team = self._team_cache.get(task_sig)
            
            if cached_team and cached_team.get_member_count() >= self.min_team_size:
                # Reuse cached team
                cached_team.last_active_at = datetime.utcnow()
                cached_team.status = TeamStatus.ACTIVE
                self._active_teams[cached_team.id] = cached_team
                return cached_team

        # Create new team
        team = AgentTeam(
            id=team_id,
            name=team_name,
            team_type=self.team_type,
            status=TeamStatus.FORMING,
        )

        # Assemble team members
        if self.agent_assembler:
            members = self.agent_assembler.assemble_team(task)
            for member in members:
                team.add_member(member)

        team.status = TeamStatus.ACTIVE
        self._active_teams[team.id] = team

        # Cache for persistent/hybrid modes
        if self.team_type in (TeamType.PERSISTENT, TeamType.HYBRID):
            task_sig = self._get_task_signature(task)
            self._team_cache.put(task_sig, team)

        return team

    def get_team(self, team_id: UUID) -> Optional[AgentTeam]:
        """Get a team by ID."""
        return self._active_teams.get(team_id)

    def list_teams(self, status: Optional[TeamStatus] = None) -> list[AgentTeam]:
        """List all active teams, optionally filtered by status."""
        teams = list(self._active_teams.values())
        
        if status:
            teams = [t for t in teams if t.status == status]
        
        return teams

    def dissolve_team(self, team_id: UUID, reason: Optional[str] = None) -> bool:
        """Dissolve a team."""
        team = self._active_teams.get(team_id)
        
        if not team:
            return False
        
        team.status = TeamStatus.DISSOLVING
        team.metadata["dissolve_reason"] = reason or "completed"
        team.metadata["dissolved_at"] = datetime.utcnow().isoformat()
        
        # Remove from active teams
        del self._active_teams[team_id]
        
        # If persistent/hybrid, keep in cache
        if self.team_type == TeamType.PERSISTENT:
            # Keep for potential reuse
            pass
        
        return True

    def add_task_to_team(self, team_id: UUID, task: Task) -> bool:
        """Add a task to an existing team."""
        team = self._active_teams.get(team_id)
        
        if not team:
            return False
        
        team.tasks.append(task.id)
        team.last_active_at = datetime.utcnow()
        
        # Re-analyze task and potentially add new members
        if self.agent_assembler:
            new_members = self.agent_assembler.assemble_team(task)
            for member in new_members:
                # Check if already in team
                existing_ids = {m.capability.agent_id for m in team.members}
                if member.capability.agent_id not in existing_ids:
                    if team.get_member_count() < self.max_team_size:
                        team.add_member(member)
        
        return True

    def get_team_stats(self) -> dict[str, Any]:
        """Get statistics about active teams."""
        return {
            "total_teams": len(self._active_teams),
            "by_status": {
                status.value: len([t for t in self._active_teams.values() if t.status == status])
                for status in TeamStatus
            },
            "by_type": {
                ttype.value: len([t for t in self._active_teams.values() if t.team_type == ttype])
                for ttype in TeamType
            },
            "cache_stats": self._team_cache.get_stats(),
        }

    def cleanup_idle_teams(self, idle_timeout: int = 300) -> int:
        """Clean up idle teams."""
        cutoff = datetime.utcnow() - timedelta(seconds=idle_timeout)
        to_remove = []
        
        for team in self._active_teams.values():
            if team.last_active_at < cutoff and team.status == TeamStatus.IDLE:
                to_remove.append(team.id)
        
        for team_id in to_remove:
            self.dissolve_team(team_id, reason="idle_timeout")
        
        return len(to_remove)

    def _get_task_signature(self, task: Task) -> str:
        """Generate a signature for task caching."""
        # Use required capabilities and description length as signature
        caps = sorted(task.required_capabilities)
        return f"{':'.join(caps)}:{len(task.description)}"

    def set_team_type(self, team_type: str) -> None:
        """Change the team type."""
        self.team_type = TeamType(team_type)
        
        # Clear cache if switching to ephemeral
        if self.team_type == TeamType.EPHEMERAL:
            self._team_cache.clear()

    def shutdown(self) -> None:
        """Shutdown the team manager and dissolve all teams."""
        for team_id in list(self._active_teams.keys()):
            self.dissolve_team(team_id, reason="shutdown")
        
        self._team_cache.clear()
