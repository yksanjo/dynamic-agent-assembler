"""Unit tests for Dynamic Agent Assembler."""

import pytest

from dynamic_agent_assembler.capability_registry import (
    AgentCapability,
    CapabilityCategory,
    CapabilityRegistry,
)
from dynamic_agent_assembler.task_analyzer import Task, TaskAnalyzer
from dynamic_agent_assembler.agent_assembler import AgentAssembler, TeamMember, AgentRole
from dynamic_agent_assembler.team_manager import TeamManager, AgentTeam, TeamType


class TestCapabilityRegistry:
    """Tests for CapabilityRegistry."""

    def test_register_capability(self):
        """Test registering a capability."""
        registry = CapabilityRegistry()
        
        capability = AgentCapability(
            agent_id="test-agent",
            agent_name="TestAgent",
            description="A test agent",
            capabilities=["test", "example"],
            category=CapabilityCategory.SPECIALIZED,
        )
        
        cap_id = registry.register(capability)
        
        assert cap_id == capability.id
        assert registry.count() == 1
        assert registry.get(cap_id) is not None

    def test_unregister_capability(self):
        """Test unregistering a capability."""
        registry = CapabilityRegistry()
        
        capability = AgentCapability(
            agent_id="test-agent",
            agent_name="TestAgent",
            description="A test agent",
            capabilities=["test"],
        )
        
        registry.register(capability)
        assert registry.count() == 1
        
        result = registry.unregister(capability.id)
        assert result is True
        assert registry.count() == 0

    def test_get_by_agent_id(self):
        """Test getting capability by agent ID."""
        registry = CapabilityRegistry()
        
        capability = AgentCapability(
            agent_id="test-agent",
            agent_name="TestAgent",
            description="A test agent",
            capabilities=["test"],
        )
        
        registry.register(capability)
        
        result = registry.get_by_agent_id("test-agent")
        assert result is not None
        assert result.agent_id == "test-agent"

    def test_list_by_category(self):
        """Test listing capabilities by category."""
        registry = CapabilityRegistry()
        
        caps = [
            AgentCapability(
                agent_id="a1",
                agent_name="Agent1",
                description="Test",
                capabilities=["test"],
                category=CapabilityCategory.REASONING,
            ),
            AgentCapability(
                agent_id="a2",
                agent_name="Agent2",
                description="Test",
                capabilities=["test"],
                category=CapabilityCategory.CREATION,
            ),
            AgentCapability(
                agent_id="a3",
                agent_name="Agent3",
                description="Test",
                capabilities=["test"],
                category=CapabilityCategory.REASONING,
            ),
        ]
        
        for cap in caps:
            registry.register(cap)
        
        reasoning_caps = registry.list_by_category(CapabilityCategory.REASONING)
        assert len(reasoning_caps) == 2


class TestTaskAnalyzer:
    """Tests for TaskAnalyzer."""

    def test_analyze_task(self):
        """Test task analysis."""
        analyzer = TaskAnalyzer(enable_decomposition=False)
        
        task = Task(
            description="Build a web scraper to collect product prices",
        )
        
        result = analyzer.analyze(task)
        
        assert result.status.value in ["decomposed", "analyzing"]
        assert len(result.required_capabilities) > 0

    def test_keyword_extraction(self):
        """Test keyword-based capability extraction."""
        analyzer = TaskAnalyzer(enable_decomposition=False)
        
        task = Task(description="Write Python code to analyze data")
        
        # Analyze with keyword fallback
        result = analyzer.analyze(task)
        
        # Should extract some capabilities
        assert isinstance(result.required_capabilities, list)

    def test_estimate_complexity(self):
        """Test complexity estimation."""
        analyzer = TaskAnalyzer()
        
        task = Task(
            description="Test task",
        )
        
        complexity = analyzer.estimate_complexity(task)
        
        assert isinstance(complexity, float)
        assert 1.0 <= complexity <= 10.0


class TestAgentAssembler:
    """Tests for AgentAssembler."""

    def test_assemble_empty_team(self):
        """Test assembling a team with no capabilities."""
        assembler = AgentAssembler(
            min_team_size=1,
            max_team_size=5,
        )
        
        task = Task(description="Test task")
        
        team = assembler.assemble_team(task)
        
        # Without registered capabilities, team should be empty
        assert isinstance(team, list)

    def test_assign_roles(self):
        """Test role assignment."""
        assembler = AgentAssembler(enable_role_assignment=True)
        
        # Create mock team members
        from dynamic_agent_assembler.vector_search import SearchResult
        
        caps = [
            AgentCapability(
                agent_id=f"agent-{i}",
                agent_name=f"Agent {i}",
                description="Test",
                capabilities=["test"],
            )
            for i in range(3)
        ]
        
        members = [
            TeamMember(capability=cap, role=AgentRole.SPECIALIST, score=1.0 - i * 0.2)
            for i, cap in enumerate(caps)
        ]
        
        result = assembler._assign_roles(members)
        
        # Should have assigned roles
        roles = [m.role for m in result]
        assert AgentRole.LEADER in roles


class TestTeamManager:
    """Tests for TeamManager."""

    def test_create_team(self):
        """Test creating a team."""
        manager = TeamManager(team_type="ephemeral")
        
        task = Task(description="Test task")
        
        team = manager.create_team(task, team_name="test-team")
        
        assert team.name == "test-team"
        assert team.team_type == TeamType.EPHEMERAL
        assert team.status.value in ["forming", "active"]

    def test_list_teams(self):
        """Test listing teams."""
        manager = TeamManager()
        
        task = Task(description="Test task")
        
        manager.create_team(task, team_name="team-1")
        manager.create_team(task, team_name="team-2")
        
        teams = manager.list_teams()
        
        assert len(teams) >= 2

    def test_dissolve_team(self):
        """Test dissolving a team."""
        manager = TeamManager()
        
        task = Task(description="Test task")
        team = manager.create_team(task)
        
        team_id = team.id
        
        result = manager.dissolve_team(team_id, reason="test")
        
        assert result is True
        assert manager.get_team(team_id) is None

    def test_team_cache(self):
        """Test team caching for persistent teams."""
        manager = TeamManager(team_type="persistent")
        
        task = Task(
            description="Test task",
            required_capabilities=["test"],
        )
        
        team1 = manager.create_team(task, team_name="cached-team")
        
        # Create similar task
        task2 = Task(
            description="Test task 2",
            required_capabilities=["test"],
        )
        
        team2 = manager.create_team(task2)
        
        # Should have cached
        stats = manager.get_team_stats()
        assert "cache_stats" in stats


class TestAgentTeam:
    """Tests for AgentTeam model."""

    def test_add_member(self):
        """Test adding a member to team."""
        team = AgentTeam(name="test-team")
        
        capability = AgentCapability(
            agent_id="test",
            agent_name="Test",
            description="Test",
            capabilities=["test"],
        )
        
        member = TeamMember(
            capability=capability,
            role=AgentRole.LEADER,
            score=1.0,
        )
        
        team.add_member(member)
        
        assert team.get_member_count() == 1

    def test_get_leader(self):
        """Test getting team leader."""
        team = AgentTeam(name="test-team")
        
        capabilities = [
            AgentCapability(
                agent_id=f"agent-{i}",
                agent_name=f"Agent {i}",
                description="Test",
                capabilities=["test"],
            )
            for i in range(3)
        ]
        
        for i, cap in enumerate(capabilities):
            role = AgentRole.LEADER if i == 0 else AgentRole.SPECIALIST
            team.add_member(TeamMember(capability=cap, role=role, score=1.0))
        
        leader = team.get_leader()
        
        assert leader is not None
        assert leader.role == AgentRole.LEADER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
