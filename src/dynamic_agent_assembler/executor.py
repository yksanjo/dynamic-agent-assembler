"""Agent Execution Engine for running tasks with assembled teams."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from dynamic_agent_assembler.task_analyzer import Task, SubTask
from dynamic_agent_assembler.team_manager import AgentTeam


class ExecutionStatus(str, Enum):
    """Status of task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionMode(str, Enum):
    """How to execute tasks with a team."""
    SEQUENTIAL = "sequential"  # One subtask at a time
    PARALLEL = "parallel"      # All subtasks at once
    HIERARCHICAL = "hierarchical"  # Leader coordinates others
    PIPELINE = "pipeline"      # Chained execution


class SubTaskResult(BaseModel):
    """Result of a subtask execution."""
    subtask_id: UUID
    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    agent_id: str
    output: Optional[str] = None


class TaskExecution(BaseModel):
    """Result of a full task execution."""
    task_id: UUID
    status: ExecutionStatus
    subtask_results: list[SubTaskResult] = Field(default_factory=list)
    final_result: Any = None
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class AgentExecutor:
    """
    Executes tasks using assembled agent teams.
    
    Supports multiple execution modes:
    - SEQUENTIAL: Process subtasks one at a time
    - PARALLEL: Process all subtasks simultaneously  
    - HIERARCHICAL: Leader delegates to team members
    - PIPELINE: Chain subtask outputs to next inputs
    """

    team: AgentTeam
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    timeout_seconds: int = 300
    retry_on_failure: bool = True
    max_retries: int = 2
    
    # Agent execution functions - these would be provided by the user
    # Map of agent_id -> async function(subtask) -> result
    _agent_handlers: dict[str, Callable] = field(default_factory=dict)
    
    def register_agent_handler(
        self, 
        agent_id: str, 
        handler: Callable[[SubTask], Any]
    ) -> None:
        """Register a handler function for an agent."""
        self._agent_handlers[agent_id] = handler

    def register_default_handler(
        self, 
        handler: Callable[[SubTask, str], Any]
    ) -> None:
        """Register a default handler for all agents."""
        self._default_handler = handler

    async def execute(self, task: Task) -> TaskExecution:
        """Execute a task with the team."""
        execution = TaskExecution(
            task_id=task.id,
            status=ExecutionStatus.RUNNING,
        )
        
        start_time = datetime.utcnow()
        
        try:
            if self.execution_mode == ExecutionMode.SEQUENTIAL:
                result = await self._execute_sequential(task)
            elif self.execution_mode == ExecutionMode.PARALLEL:
                result = await self._execute_parallel(task)
            elif self.execution_mode == ExecutionMode.HIERARCHICAL:
                result = await self._execute_hierarchical(task)
            elif self.execution_mode == ExecutionMode.PIPELINE:
                result = await self._execute_pipeline(task)
            else:
                result = await self._execute_sequential(task)
            
            execution.subtask_results = result
            execution.status = ExecutionStatus.COMPLETED
            
        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.errors.append(str(e))
        
        end_time = datetime.utcnow()
        execution.completed_at = end_time
        execution.duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Aggregate final result
        execution.final_result = self._aggregate_results(execution.subtask_results)
        
        return execution

    async def _execute_sequential(self, task: Task) -> list[SubTaskResult]:
        """Execute subtasks one at a time."""
        results = []
        
        for subtask in task.subtasks:
            result = await self._execute_single_subtask(subtask)
            results.append(result)
            
            # Stop on failure if no retry
            if result.status == ExecutionStatus.FAILED and not self.retry_on_failure:
                break
        
        return results

    async def _execute_parallel(self, task: Task) -> list[SubTaskResult]:
        """Execute all subtasks in parallel."""
        if not task.subtasks:
            return []
        
        # Create tasks for all subtasks
        subtasks = task.subtasks
        tasks = [self._execute_single_subtask(subtask) for subtask in subtasks]
        
        # Execute all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(SubTaskResult(
                    subtask_id=subtasks[i].id,
                    status=ExecutionStatus.FAILED,
                    error=str(result),
                    agent_id="unknown",
                ))
            else:
                processed_results.append(result)
        
        return processed_results

    async def _execute_hierarchical(self, task: Task) -> list[SubTaskResult]:
        """Execute with team leader coordinating."""
        leader = self.team.get_leader()
        
        if not leader:
            return await self._execute_sequential(task)
        
        results = []
        
        # Leader processes tasks and delegates
        for subtask in task.subtasks:
            # Find best agent for this subtask
            best_agent = self._find_best_agent(subtask)
            
            if best_agent:
                result = await self._execute_with_agent(subtask, best_agent)
            else:
                # Fallback to sequential
                result = await self._execute_single_subtask(subtask)
            
            results.append(result)
        
        return results

    async def _execute_pipeline(self, task: Task) -> list[SubTaskResult]:
        """Execute subtasks in pipeline (output feeds input)."""
        results = []
        context = {}  # Shared context between subtasks
        
        for subtask in task.subtasks:
            # Inject previous results into context
            if results:
                subtask.metadata["pipeline_context"] = context
                subtask.metadata["previous_results"] = [r.output for r in results]
            
            result = await self._execute_single_subtask(subtask)
            results.append(result)
            
            # Update context with output
            if result.output:
                context[subtask.id] = result.output
            
            # Stop on failure
            if result.status == ExecutionStatus.FAILED:
                break
        
        return results

    async def _execute_single_subtask(self, subtask: SubTask) -> SubTaskResult:
        """Execute a single subtask."""
        # Find best agent for this subtask
        agent = self._find_best_agent(subtask)
        
        if not agent:
            return SubTaskResult(
                subtask_id=subtask.id,
                status=ExecutionStatus.FAILED,
                error="No suitable agent found",
                agent_id="none",
            )
        
        return await self._execute_with_agent(subtask, agent)

    async def _execute_with_agent(
        self, 
        subtask: SubTask, 
        agent_id: str
    ) -> SubTaskResult:
        """Execute subtask with a specific agent."""
        result = SubTaskResult(
            subtask_id=subtask.id,
            status=ExecutionStatus.RUNNING,
            agent_id=agent_id,
        )
        
        handler = self._agent_handlers.get(agent_id)
        
        try:
            if handler:
                # Use registered handler
                output = await asyncio.wait_for(
                    handler(subtask),
                    timeout=self.timeout_seconds
                )
            elif hasattr(self, '_default_handler') and self._default_handler:
                # Use default handler
                output = await asyncio.wait_for(
                    self._default_handler(subtask, agent_id),
                    timeout=self.timeout_seconds
                )
            else:
                # No handler - simulate execution
                output = await self._simulate_execution(subtask, agent_id)
            
            result.status = ExecutionStatus.COMPLETED
            result.output = str(output) if output else None
            result.result = output
            
        except asyncio.TimeoutError:
            result.status = ExecutionStatus.FAILED
            result.error = f"Timeout after {self.timeout_seconds}s"
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
        
        result.completed_at = datetime.utcnow()
        if result.started_at and result.completed_at:
            result.duration_ms = int(
                (result.completed_at - result.started_at).total_seconds() * 1000
            )
        
        return result

    async def _simulate_execution(
        self, 
        subtask: SubTask, 
        agent_id: str
    ) -> str:
        """Simulate execution when no handler is provided."""
        await asyncio.sleep(0.1)  # Simulate work
        
        agent = next(
            (m for m in self.team.members if m.capability.agent_id == agent_id),
            None
        )
        
        agent_name = agent.capability.agent_name if agent else agent_id
        
        return f"[{agent_name}] executed: {subtask.description}"

    def _find_best_agent(self, subtask: SubTask) -> Optional[str]:
        """Find the best agent for a subtask based on capabilities."""
        if not self.team.members:
            return None
        
        required = set(subtask.required_capabilities)
        
        best_agent = None
        best_score = 0
        
        for member in self.team.members:
            agent_caps = set(member.capability.capabilities)
            overlap = len(required & agent_caps)
            
            if overlap > best_score:
                best_score = overlap
                best_agent = member.capability.agent_id
        
        return best_agent

    def _aggregate_results(
        self, 
        results: list[SubTaskResult]
    ) -> dict[str, Any]:
        """Aggregate results from all subtasks."""
        successful = [r for r in results if r.status == ExecutionStatus.COMPLETED]
        failed = [r for r in results if r.status == ExecutionStatus.FAILED]
        
        return {
            "total_subtasks": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "outputs": [r.output for r in successful if r.output],
            "errors": [r.error for r in failed if r.error],
        }


class ExecutionContext:
    """Context for executing multiple tasks with the same team."""

    def __init__(self, executor: AgentExecutor):
        self.executor = executor
        self.executions: dict[UUID, TaskExecution] = {}

    async def execute_task(self, task: Task) -> TaskExecution:
        """Execute a task and track it."""
        execution = await self.executor.execute(task)
        self.executions[task.id] = execution
        return execution

    def get_execution(self, task_id: UUID) -> Optional[TaskExecution]:
        """Get execution by task ID."""
        return self.executions.get(task_id)

    def list_executions(self) -> list[TaskExecution]:
        """List all executions."""
        return list(self.executions.values())

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all executions."""
        if not self.executions:
            return {"total": 0}
        
        total = len(self.executions)
        completed = sum(
            1 for e in self.executions.values() 
            if e.status == ExecutionStatus.COMPLETED
        )
        failed = sum(
            1 for e in self.executions.values() 
            if e.status == ExecutionStatus.FAILED
        )
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total if total > 0 else 0,
        }
