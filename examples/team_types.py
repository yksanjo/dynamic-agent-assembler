"""Example demonstrating different team types (Ephemeral, Persistent, Hybrid)."""

import time
from dynamic_agent_assembler import DynamicAgentAssembler


def register_sample_agents(assembler):
    """Register sample agents for the example."""
    agents = [
        {
            "agent_id": "code-gen",
            "agent_name": "CodeGenerator",
            "description": "Expert in generating high-quality code",
            "capabilities": ["code generation", "refactoring"],
            "category": "creation",
        },
        {
            "agent_id": "data-analyst",
            "agent_name": "DataAnalyzer",
            "description": "Expert in data analysis and visualization",
            "capabilities": ["data analysis", "visualization"],
            "category": "analysis",
        },
        {
            "agent_id": "research-agent",
            "agent_name": "ResearchAssistant",
            "description": "Expert in research and investigation",
            "capabilities": ["research", "web scraping"],
            "category": "analysis",
        },
    ]
    
    for agent in agents:
        assembler.register_agent(**agent)


def example_ephemeral_teams():
    """Example: Ephemeral Teams for one-off tasks."""
    print("\n" + "=" * 50)
    print("EPHEMERAL TEAMS EXAMPLE")
    print("=" * 50)
    
    assembler = DynamicAgentAssembler()
    register_sample_agents(assembler)
    
    # Each task gets a new team that is dissolved after
    tasks = [
        "Write a Python script to process CSV files",
        "Analyze customer feedback and create a report",
        "Research competitor pricing strategies",
    ]
    
    teams = []
    for i, task_desc in enumerate(tasks, 1):
        print(f"\n[Ephemeral Task {i}] {task_desc}")
        
        # Analyze and build team
        task = assembler.analyze_task(task_desc)
        team = assembler.build_team(task, team_type="ephemeral")
        
        print(f"  Team: {team.name}")
        print(f"  Members: {team.get_member_count()}")
        for member in team.members:
            print(f"    - {member.capability.agent_name}")
        
        teams.append(team)
        
        # Immediately dissolve (simulating task completion)
        assembler.dissolve_team(str(team.id), reason="task_completed")
        print(f"  Team dissolved after task completion")
    
    print(f"\n[Ephemeral Summary]")
    print(f"  Total teams created: {len(teams)}")
    print(f"  Active teams: {len(assembler.list_teams())}")
    
    assembler.shutdown()


def example_persistent_teams():
    """Example: Persistent Teams for recurring tasks."""
    print("\n" + "=" * 50)
    print("PERSISTENT TEAMS EXAMPLE")
    print("=" * 50)
    
    assembler = DynamicAgentAssembler()
    register_sample_agents(assembler)
    
    # Create a persistent team for recurring analysis tasks
    task_desc = "Weekly sales data analysis and dashboard"
    task = assembler.analyze_task(task_desc)
    
    print(f"\n[Creating Persistent Team]")
    team = assembler.build_team(task, team_type="persistent", team_name="sales-team")
    
    print(f"  Team: {team.name}")
    print(f"  Type: {team.team_type.value}")
    print(f"  Members: {team.get_member_count()}")
    
    # Simulate multiple task executions with the same team
    print(f"\n[Executing multiple related tasks]")
    
    related_tasks = [
        "Analyze this week's sales numbers",
        "Compare with last week's performance",
        "Generate monthly summary report",
    ]
    
    for i, related_task in enumerate(related_tasks, 1):
        print(f"\n  Task {i}: {related_task}")
        
        # Analyze new task
        new_task = assembler.analyze_task(related_task)
        
        # Add to existing team
        assembler.add_task_to_team(str(team.id), new_task)
        
        print(f"    Team still active: {team.is_active()}")
        print(f"    Total tasks: {len(team.tasks)}")
    
    print(f"\n[Persistent Summary]")
    print(f"  Team reused: {len(related_tasks)} times")
    print(f"  Active teams: {len(assembler.list_teams())}")
    
    # Cleanup
    assembler.dissolve_team(str(team.id), reason="project_completed")
    assembler.shutdown()


def example_hybrid_teams():
    """Example: Hybrid Teams balancing reuse and flexibility."""
    print("\n" + "=" * 50)
    print("HYBRID TEAMS EXAMPLE")
    print("=" * 50)
    
    assembler = DynamicAgentAssembler()
    register_sample_agents(assembler)
    
    # Hybrid mode: tries to reuse teams when beneficial
    # but creates new ones when needed
    
    print(f"\n[Hybrid Team Strategy]")
    print("  - First task: Creates new team")
    print("  - Similar task: Tries to reuse cached team")
    print("  - Different task: Creates new team")
    
    # Task group 1: Code generation tasks
    code_tasks = [
        "Generate a REST API in Python",
        "Write unit tests for the API",
    ]
    
    print(f"\n[Task Group 1: Code Generation]")
    team1 = assembler.build_team_from_description(
        description=code_tasks[0],
        team_type="hybrid",
        team_name="code-team",
    )
    print(f"  Created: {team1.name}")
    
    # Similar task should try to reuse
    team1_reuse = assembler.build_team_from_description(
        description=code_tasks[1],
        team_type="hybrid",
    )
    print(f"  Reused: {team1_reuse.name} (same: {team1.id == team1_reuse.id})")
    
    # Task group 2: Different domain (should create new team)
    analysis_task = "Analyze customer churn rate"
    team2 = assembler.build_team_from_description(
        description=analysis_task,
        team_type="hybrid",
    )
    print(f"\n[Task Group 2: Analysis]")
    print(f"  Created new: {team2.name} (different: {team1.id != team2.id})")
    
    print(f"\n[Hybrid Summary]")
    print(f"  Active teams: {len(assembler.list_teams())}")
    
    # Cleanup
    assembler.shutdown()


def example_team_selection_strategies():
    """Example: Different team selection strategies."""
    print("\n" + "=" * 50)
    print("SELECTION STRATEGIES EXAMPLE")
    print("=" * 50)
    
    from dynamic_agent_assembler.config import Config
    
    strategies = ["semantic_similarity", "weighted", "greedy"]
    
    for strategy in strategies:
        print(f"\n[Strategy: {strategy}]")
        
        # Create config with specific strategy
        config = Config()
        config.team_assembly.selection_strategy = strategy
        
        assembler = DynamicAgentAssembler(config)
        register_sample_agents(assembler)
        
        task = assembler.analyze_task(
            "Build a data pipeline with API integration"
        )
        
        team = assembler.build_team(task, team_type="ephemeral")
        
        print(f"  Team members: {team.get_member_count()}")
        for member in team.members:
            print(f"    - {member.capability.agent_name} ({member.role.value})")
        
        assembler.shutdown()


def main():
    """Run all team type examples."""
    print("=" * 60)
    print("Dynamic Agent Assembler - Team Types Examples")
    print("=" * 60)
    
    example_ephemeral_teams()
    example_persistent_teams()
    example_hybrid_teams()
    example_team_selection_strategies()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
