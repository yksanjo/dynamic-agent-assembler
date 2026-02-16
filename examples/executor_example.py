"""Example demonstrating the Agent Execution Engine."""

import asyncio
from dynamic_agent_assembler import (
    DynamicAgentAssembler,
    AgentExecutor,
    ExecutionMode,
    ExecutionContext,
)


def register_sample_agents(assembler):
    """Register sample agents for the example."""
    agents = [
        {
            "agent_id": "code-gen",
            "agent_name": "CodeGenerator",
            "description": "Expert in generating code",
            "capabilities": ["code generation", "refactoring"],
            "category": "creation",
        },
        {
            "agent_id": "data-analyst",
            "agent_name": "DataAnalyzer",
            "description": "Expert in data analysis",
            "capabilities": ["data analysis", "visualization"],
            "category": "analysis",
        },
        {
            "agent_id": "research-agent",
            "agent_name": "ResearchAssistant",
            "description": "Expert in research",
            "capabilities": ["research", "web scraping"],
            "category": "analysis",
        },
    ]
    
    for agent in agents:
        assembler.register_agent(**agent)


async def example_basic_execution():
    """Example: Basic task execution with a team."""
    print("\n" + "=" * 50)
    print("BASIC EXECUTION EXAMPLE")
    print("=" * 50)
    
    assembler = DynamicAgentAssembler()
    register_sample_agents(assembler)
    
    # Build a team
    task = assembler.analyze_task(
        "Build a data pipeline that scrapes data, processes it, and generates insights"
    )
    team = assembler.build_team(task, team_type="ephemeral", team_name="data-pipeline-team")
    
    print(f"\n[Team Created]")
    print(f"  Name: {team.name}")
    print(f"  Members: {team.get_member_count()}")
    
    # Create executor
    executor = AgentExecutor(
        team=team,
        execution_mode=ExecutionMode.SEQUENTIAL,
    )
    
    # Execute the task
    print(f"\n[Executing Task]")
    result = await executor.execute(task)
    
    print(f"  Status: {result.status.value}")
    print(f"  Duration: {result.duration_ms}ms")
    print(f"  Subtasks: {len(result.subtask_results)}")
    
    for subtask_result in result.subtask_results:
        print(f"    - {subtask_result.agent_id}: {subtask_result.status.value}")
        if subtask_result.output:
            print(f"      Output: {subtask_result.output[:50]}...")
    
    print(f"\n[Final Result]")
    print(f"  Total: {result.final_result['total_subtasks']}")
    print(f"  Successful: {result.final_result['successful']}")
    print(f"  Failed: {result.final_result['failed']}")
    
    assembler.shutdown()


async def example_parallel_execution():
    """Example: Parallel task execution."""
    print("\n" + "=" * 50)
    print("PARALLEL EXECUTION EXAMPLE")
    print("=" * 50)
    
    assembler = DynamicAgentAssembler()
    register_sample_agents(assembler)
    
    # Build a team
    task = assembler.analyze_task(
        "Research market trends, analyze competitor data, and write a report"
    )
    team = assembler.build_team(task, team_type="ephemeral")
    
    # Create executor with parallel mode
    executor = AgentExecutor(
        team=team,
        execution_mode=ExecutionMode.PARALLEL,
    )
    
    print(f"\n[Executing in Parallel]")
    result = await executor.execute(task)
    
    print(f"  Status: {result.status.value}")
    print(f"  Duration: {result.duration_ms}ms")
    
    # Show parallel execution benefits
    sequential_time = sum(r.duration_ms or 0 for r in result.subtask_results)
    print(f"  Sequential time would be: ~{sequential_time}ms")
    print(f"  Actual parallel time: {result.duration_ms}ms")
    
    assembler.shutdown()


async def example_custom_handlers():
    """Example: Custom agent handlers."""
    print("\n" + "=" * 50)
    print("CUSTOM HANDLERS EXAMPLE")
    print("=" * 50)
    
    assembler = DynamicAgentAssembler()
    register_sample_agents(assembler)
    
    task = assembler.analyze_task("Generate Python code for data processing")
    team = assembler.build_team(task, team_type="ephemeral")
    
    # Create executor with custom handlers
    executor = AgentExecutor(team=team)
    
    # Register custom handlers for each agent
    async def code_gen_handler(subtask):
        # Simulate code generation
        await asyncio.sleep(0.2)
        return f"Generated Python code for: {subtask.description}"
    
    async def data_handler(subtask):
        await asyncio.sleep(0.15)
        return f"Analyzed data for: {subtask.description}"
    
    executor.register_agent_handler("code-gen", code_gen_handler)
    executor.register_agent_handler("data-analyst", data_handler)
    
    print(f"\n[Executing with Custom Handlers]")
    result = await executor.execute(task)
    
    print(f"  Status: {result.status.value}")
    for r in result.subtask_results:
        print(f"    {r.agent_id}: {r.output}")
    
    assembler.shutdown()


async def example_execution_context():
    """Example: Execution context for multiple tasks."""
    print("\n" + "=" * 50)
    print("EXECUTION CONTEXT EXAMPLE")
    print("=" * 50)
    
    assembler = DynamicAgentAssembler()
    register_sample_agents(assembler)
    
    team = assembler.build_team(
        assembler.analyze_task("Data analysis tasks"),
        team_type="persistent",
    )
    
    executor = AgentExecutor(team=team)
    context = ExecutionContext(executor)
    
    # Execute multiple tasks
    tasks = [
        "Analyze Q1 sales data",
        "Analyze Q2 sales data", 
        "Analyze Q3 sales data",
        "Generate quarterly report",
    ]
    
    print(f"\n[Executing Multiple Tasks]")
    for task_desc in tasks:
        task = assembler.analyze_task(task_desc)
        execution = await context.execute_task(task)
        print(f"  {task_desc}: {execution.status.value}")
    
    # Get summary
    summary = context.get_summary()
    print(f"\n[Summary]")
    print(f"  Total: {summary['total']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    
    assembler.shutdown()


async def example_pipeline_execution():
    """Example: Pipeline execution mode."""
    print("\n" + "=" * 50)
    print("PIPELINE EXECUTION EXAMPLE")
    print("=" * 50)
    
    assembler = DynamicAgentAssembler()
    register_sample_agents(assembler)
    
    task = assembler.analyze_task(
        "Extract data from API, transform it, and load to database"
    )
    team = assembler.build_team(task, team_type="ephemeral")
    
    executor = AgentExecutor(
        team=team,
        execution_mode=ExecutionMode.PIPELINE,
    )
    
    print(f"\n[Executing in Pipeline Mode]")
    result = await executor.execute(task)
    
    print(f"  Status: {result.status.value}")
    print(f"  Duration: {result.duration_ms}ms")
    
    # Show how outputs flow between subtasks
    print(f"\n[Pipeline Flow]")
    for r in result.subtask_results:
        print(f"  Step {r.subtask_id}: {r.output[:60] if r.output else 'N/A'}...")
    
    assembler.shutdown()


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Agent Execution Engine Examples")
    print("=" * 60)
    
    await example_basic_execution()
    await example_parallel_execution()
    await example_custom_handlers()
    await example_execution_context()
    await example_pipeline_execution()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
