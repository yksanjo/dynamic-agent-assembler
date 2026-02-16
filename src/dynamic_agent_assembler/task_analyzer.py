"""Task Analyzer module with LLM-based reasoning for multi-stage task processing."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Task status."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    DECOMPOSED = "decomposed"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SubTask(BaseModel):
    """Represents a subtask derived from task analysis."""
    id: UUID = Field(default_factory=uuid4)
    description: str
    required_capabilities: list[str] = Field(default_factory=list)
    dependencies: list[UUID] = Field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_complexity: float = 1.0  # 1-10 scale
    confidence: float = 1.0  # 0-1 scale
    assigned_agent_id: Optional[str] = None


class Task(BaseModel):
    """Represents a task to be executed by an agent team."""
    id: UUID = Field(default_factory=uuid4)
    description: str
    context: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    subtasks: list[SubTask] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_subtask(self, subtask: SubTask) -> None:
        """Add a subtask to the task."""
        self.subtasks.append(subtask)
        self.updated_at = datetime.utcnow()

    def get_execution_order(self) -> list[list[SubTask]]:
        """Get subtasks grouped by execution order (parallel within groups)."""
        if not self.subtasks:
            return []
        
        # Simple topological sort
        remaining = set(s.id for s in self.subtasks)
        groups = []
        
        while remaining:
            # Find tasks with no remaining dependencies
            ready = []
            for subtask in self.subtasks:
                if subtask.id in remaining:
                    deps = set(subtask.dependencies) & remaining
                    if not deps:
                        ready.append(subtask)
            
            if not ready:
                # Circular dependency - just add remaining
                ready = [s for s in self.subtasks if s.id in remaining]
            
            groups.append(ready)
            for s in ready:
                remaining.discard(s.id)
        
        return groups


class TaskAnalyzer:
    """Analyzes tasks using LLM-based reasoning for capability matching."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        enable_decomposition: bool = True,
        max_subtasks: int = 10,
        confidence_threshold: float = 0.7,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_decomposition = enable_decomposition
        self.max_subtasks = max_subtasks
        self.confidence_threshold = confidence_threshold
        self._llm_client = None

    def _get_llm_client(self):
        """Get or initialize LLM client."""
        if self._llm_client is not None:
            return self._llm_client
        
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._llm_client = OpenAI()
            except ImportError:
                return None
        elif self.provider == "anthropic":
            try:
                import anthropic
                self._llm_client = anthropic.Anthropic()
            except ImportError:
                return None
        
        return self._llm_client

    def analyze(self, task: Task) -> Task:
        """Analyze a task and decompose if needed."""
        task.status = TaskStatus.ANALYZING
        
        # Extract required capabilities from task description
        capabilities = self._extract_capabilities(task.description)
        task.required_capabilities = capabilities
        
        if self.enable_decomposition:
            # Decompose into subtasks
            subtasks = self._decompose_task(task)
            task.subtasks = [s for s in subtasks if s.confidence >= self.confidence_threshold]
        
        task.status = TaskStatus.DECOMPOSED
        task.updated_at = datetime.utcnow()
        
        return task

    def _extract_capabilities(self, description: str) -> list[str]:
        """Extract required capabilities from task description."""
        # Use LLM if available
        client = self._get_llm_client()
        
        if client:
            try:
                if self.provider == "openai":
                    response = client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[
                            {
                                "role": "system",
                                "content": """Extract key capabilities required to accomplish this task. 
                                Return a comma-separated list of capabilities (e.g., "code generation, 
                                data analysis, web scraping"). Focus on skills and expertise areas."""
                            },
                            {
                                "role": "user",
                                "content": description
                            }
                        ]
                    )
                    capabilities_text = response.choices[0].message.content
                    return [c.strip() for c in capabilities_text.split(",") if c.strip()]
                
                elif self.provider == "anthropic":
                    response = client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system="""Extract key capabilities required to accomplish this task. 
                        Return a comma-separated list of capabilities (e.g., "code generation, 
                        data analysis, web scraping"). Focus on skills and expertise areas.""",
                        messages=[
                            {"role": "user", "content": description}
                        ]
                    )
                    capabilities_text = response.content[0].text
                    return [c.strip() for c in capabilities_text.split(",") if c.strip()]
            except Exception:
                pass
        
        # Fallback: keyword-based extraction
        return self._keyword_based_extraction(description)

    def _keyword_based_extraction(self, description: str) -> list[str]:
        """Extract capabilities using keyword matching."""
        description_lower = description.lower()
        
        capability_keywords = {
            "code generation": ["code", "generate", "implement", "build", "create"],
            "data analysis": ["analyze", "data", "statistics", "insights", "metrics"],
            "research": ["research", "investigate", "find", "search", "explore"],
            "writing": ["write", "draft", "compose", "edit", "文案"],
            "translation": ["translate", "localize", "language"],
            "web scraping": ["scrape", "crawl", "extract", "web"],
            "api integration": ["api", "integrate", "connect", "endpoint"],
            "testing": ["test", "validate", "verify", "qa"],
            "debugging": ["debug", "fix", "troubleshoot", "error"],
            "optimization": ["optimize", "improve", "performance", "efficient"],
            "documentation": ["document", "docs", "specification"],
            "design": ["design", "ui", "ux", "interface", "visual"],
            "project management": ["manage", "plan", "coordinate", "organize"],
        }
        
        capabilities = []
        for capability, keywords in capability_keywords.items():
            if any(kw in description_lower for kw in keywords):
                capabilities.append(capability)
        
        # If no capabilities found, add a default
        if not capabilities:
            capabilities.append("general assistance")
        
        return capabilities

    def _decompose_task(self, task: Task) -> list[SubTask]:
        """Decompose a task into subtasks using LLM reasoning."""
        client = self._get_llm_client()
        
        if client:
            try:
                if self.provider == "openai":
                    response = client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[
                            {
                                "role": "system",
                                "content": f"""Decompose this task into {self.max_subtasks} or fewer subtasks.
                                For each subtask, provide:
                                - description: What the subtask does
                                - required_capabilities: Comma-separated list of capabilities needed
                                - estimated_complexity: A number 1-10
                                - confidence: A number 0-1 indicating how confident you are in this decomposition
                                
                                Format each subtask on a new line as:
                                SUBTASK: <description> | CAPABILITIES: <capabilities> | COMPLEXITY: <complexity> | CONFIDENCE: <confidence>"""
                            },
                            {
                                "role": "user",
                                "content": f"{task.description}\n\nContext: {task.context or 'None'}"
                            }
                        ]
                    )
                    return self._parse_llm_decomposition(response.choices[0].message.content)
                
                elif self.provider == "anthropic":
                    response = client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system=f"""Decompose this task into {self.max_subtasks} or fewer subtasks.
                        For each subtask, provide:
                        - description: What the subtask does
                        - required_capabilities: Comma-separated list of capabilities needed
                        - estimated_complexity: A number 1-10
                        - confidence: A number 0-1 indicating how confident you are in this decomposition
                        
                        Format each subtask on a new line as:
                        SUBTASK: <description> | CAPABILITIES: <capabilities> | COMPLEXITY: <complexity> | CONFIDENCE: <confidence>""",
                        messages=[
                            {"role": "user", "content": f"{task.description}\n\nContext: {task.context or 'None'}"}
                        ]
                    )
                    return self._parse_llm_decomposition(response.content[0].text)
            except Exception:
                pass
        
        # Fallback: simple decomposition
        return self._simple_decomposition(task)

    def _parse_llm_decomposition(self, text: str) -> list[SubTask]:
        """Parse LLM response into SubTask objects."""
        subtasks = []
        
        for line in text.strip().split("\n"):
            if "SUBTASK:" not in line:
                continue
            
            try:
                # Parse line format
                parts = line.split("|")
                description = ""
                capabilities = []
                complexity = 5.0
                confidence = 0.8
                
                for part in parts:
                    part = part.strip()
                    if part.startswith("SUBTASK:"):
                        description = part.replace("SUBTASK:", "").strip()
                    elif part.startswith("CAPABILITIES:"):
                        caps = part.replace("CAPABILITIES:", "").strip()
                        capabilities = [c.strip() for c in caps.split(",") if c.strip()]
                    elif part.startswith("COMPLEXITY:"):
                        complexity = float(part.replace("COMPLEXITY:", "").strip())
                    elif part.startswith("CONFIDENCE:"):
                        confidence = float(part.replace("CONFIDENCE:", "").strip())
                
                if description:
                    subtasks.append(SubTask(
                        description=description,
                        required_capabilities=capabilities,
                        estimated_complexity=complexity,
                        confidence=confidence,
                    ))
            except Exception:
                continue
        
        return subtasks

    def _simple_decomposition(self, task: Task) -> list[SubTask]:
        """Simple fallback decomposition."""
        return [
            SubTask(
                description=task.description,
                required_capabilities=task.required_capabilities,
                estimated_complexity=5.0,
                confidence=0.5,
            )
        ]

    def estimate_complexity(self, task: Task) -> float:
        """Estimate overall task complexity."""
        if not task.subtasks:
            return 5.0
        
        # Weighted average based on complexity and confidence
        total_weight = 0
        weighted_sum = 0
        
        for subtask in task.subtasks:
            weight = subtask.confidence
            weighted_sum += subtask.estimated_complexity * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 5.0
