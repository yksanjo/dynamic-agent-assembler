"""
Dynamic Agent Assembler - Runtime Agent Team Construction

A framework for building fit-for-purpose agent teams dynamically based on
task-specific requirements using vector-based capability matching.
"""

from dynamic_agent_assembler.assembler import DynamicAgentAssembler
from dynamic_agent_assembler.config import Config
from dynamic_agent_assembler.capability_registry import AgentCapability, CapabilityCategory, CapabilityRegistry
from dynamic_agent_assembler.vector_search import VectorSearchEngine, SearchResult
from dynamic_agent_assembler.task_analyzer import TaskAnalyzer, Task, SubTask, TaskPriority, TaskStatus
from dynamic_agent_assembler.agent_assembler import AgentAssembler, AgentRole, TeamMember, TeamStrategy
from dynamic_agent_assembler.team_manager import TeamManager, AgentTeam, TeamType, TeamStatus as TeamStatusEnum, TeamCache

__version__ = "0.1.0"

__all__ = [
    "DynamicAgentAssembler",
    "Config",
    "AgentCapability",
    "CapabilityCategory",
    "CapabilityRegistry",
    "VectorSearchEngine",
    "SearchResult",
    "TaskAnalyzer",
    "Task",
    "SubTask",
    "TaskPriority",
    "TaskStatus",
    "AgentAssembler",
    "AgentRole",
    "TeamMember",
    "TeamStrategy",
    "TeamManager",
    "AgentTeam",
    "TeamType",
    "TeamStatusEnum",
    "TeamCache",
]
