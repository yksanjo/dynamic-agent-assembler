# Dynamic Agent Assembler

Runtime Agent Team Construction with Vector-based Capability Matching

## Overview

The Dynamic Agent Assembler is a framework for building fit-for-purpose agent teams dynamically based on task-specific requirements. It uses vector embeddings for semantic capability matching, enabling precise team composition without manual configuration.

## Features

- **Vector Search Integration**: ChromaDB + sentence-transformers for semantic capability matching
- **Task Analysis**: LLM-based reasoning for multi-stage task decomposition
- **Flexible Team Types**:
  - **Ephemeral Teams**: Single task execution, immediate teardown for diverse, unrelated tasks
  - **Persistent Teams**: Multiple related tasks with incremental adaptation for recurring patterns
  - **Hybrid Approach**: Dynamic optimization between construction overhead and reuse benefit
- **Multiple Selection Strategies**: Semantic similarity, weighted, greedy, and ensemble approaches
- **Role Assignment**: Automatic assignment of Leader, Coordinator, Specialist, Executor, and Reviewer roles
- **CLI Interface**: Full command-line interface for agent and team management

## Installation

```bash
# Install package
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install with OpenAI support
pip install -e ".[openai]"

# Install with Anthropic support
pip install -e ".[anthropic]"
```

## Quick Start

```python
from dynamic_agent_assembler import DynamicAgentAssembler

# Initialize the assembler
assembler = DynamicAgentAssembler()

# Register agents with capabilities
assembler.register_agent(
    agent_id="agent-001",
    agent_name="CodeGenerator",
    description="Expert in generating high-quality code",
    capabilities=["code generation", "code review", "refactoring"],
    category="creation",
    keywords=["python", "javascript", "typescript"],
)

assembler.register_agent(
    agent_id="agent-002",
    agent_name="DataAnalyzer",
    description="Expert in data analysis and visualization",
    capabilities=["data analysis", "statistics", "visualization"],
    category="analysis",
    keywords=["pandas", "numpy", "matplotlib"],
)

# Build a team for a task
team = assembler.build_team_from_description(
    description="Analyze user data and generate a report with visualizations",
    team_type="ephemeral",
)

print(f"Built team: {team.name}")
print(f"Members: {team.get_member_count()}")
for member in team.members:
    print(f"  - {member.capability.agent_name} ({member.role.value})")
```

## CLI Usage

```bash
# Register an agent
agent-assembler register \
  --agent-id "agent-001" \
  --name "CodeGenerator" \
  --description "Expert in generating code" \
  --capabilities "code generation,code review,refactoring" \
  --category "creation"

# List registered agents
agent-assembler list-agents

# Search for agents
agent-assembler search "data analysis"

# Analyze a task
agent-assembler analyze "Build a web scraper to collect product prices"

# Build a team
agent-assembler build-team "Analyze sales data and create visualizations"

# List active teams
agent-assembler list-teams

# Show statistics
agent-assembler stats
```

## Configuration

Create a `config.yaml` file to customize behavior:

```yaml
vector_search:
  embedding_model:
    name: "sentence-transformers/all-MiniLM-L6-v2"
    device: "cpu"
  chromadb:
    persist_directory: "./data/chromadb"
    collection_name: "agent_capabilities"

task_analysis:
  provider: "openai"  # or "anthropic" or "mock"
  model: "gpt-4"
  temperature: 0.7

team_assembly:
  default_team_type: "ephemeral"
  min_team_size: 1
  max_team_size: 10
  optimal_team_size: 3
```

## Architecture

### Components

1. **VectorSearchEngine**: ChromaDB + sentence-transformers for semantic search
2. **CapabilityRegistry**: Manages agent capabilities with vector indexing
3. **TaskAnalyzer**: LLM-based task decomposition
4. **AgentAssembler**: Team construction based on capability matching
5. **TeamManager**: Lifecycle management for ephemeral/persistent/hybrid teams

### Capability Matching Process

1. **Task Analysis**: Multi-stage processing with LLM-based reasoning
2. **Capability Search**: Vector embeddings with semantic similarity
3. **Team Assembly**: Dynamic construction based on matching scores

## Examples

See the `examples/` directory for more detailed examples:

- `basic_usage.py`: Basic usage patterns
- `team_types.py`: Ephemeral, persistent, and hybrid team examples
- `custom_agents.py`: Registering custom agents with specific capabilities

## License

MIT
