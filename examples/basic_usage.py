"""Basic usage example for Dynamic Agent Assembler."""

from dynamic_agent_assembler import DynamicAgentAssembler


def main():
    """Run basic usage example."""
    print("=" * 60)
    print("Dynamic Agent Assembler - Basic Usage Example")
    print("=" * 60)
    
    # Initialize the assembler
    print("\n[1] Initializing assembler...")
    assembler = DynamicAgentAssembler()
    
    # Register sample agents
    print("\n[2] Registering agents...")
    
    agents = [
        {
            "agent_id": "code-gen",
            "agent_name": "CodeGenerator",
            "description": "Expert in generating high-quality code across multiple languages",
            "capabilities": ["code generation", "code review", "refactoring", "testing"],
            "category": "creation",
            "keywords": ["python", "javascript", "typescript", "java"],
        },
        {
            "agent_id": "data-analyst",
            "agent_name": "DataAnalyzer",
            "description": "Expert in data analysis, statistics, and visualization",
            "capabilities": ["data analysis", "statistics", "visualization", "reporting"],
            "category": "analysis",
            "keywords": ["pandas", "numpy", "matplotlib", "tableau"],
        },
        {
            "agent_id": "research-agent",
            "agent_name": "ResearchAssistant",
            "description": "Expert in research, investigation, and information gathering",
            "capabilities": ["research", "investigation", "web scraping", "summarization"],
            "category": "analysis",
            "keywords": ["search", "scrape", "summarize", "facts"],
        },
        {
            "agent_id": "writer-agent",
            "agent_name": "ContentWriter",
            "description": "Expert in writing, editing, and content creation",
            "capabilities": ["writing", "editing", "copywriting", "translation"],
            "category": "creation",
            "keywords": ["copy", "文案", "translate", "edit"],
        },
        {
            "agent_id": "api-agent",
            "agent_name": "APIIntegrator",
            "description": "Expert in API integration and web services",
            "capabilities": ["api integration", "web services", "rest", "graphql"],
            "category": "execution",
            "keywords": ["api", "integration", "endpoint", "webhook"],
        },
    ]
    
    for agent in agents:
        capability = assembler.register_agent(**agent)
        print(f"  - Registered: {capability.agent_name}")
    
    # List agents
    print("\n[3] Registered agents:")
    all_agents = assembler.list_agents()
    for agent in all_agents:
        print(f"  - {agent.agent_name}: {', '.join(agent.capabilities)}")
    
    # Search for agents
    print("\n[4] Searching for 'data analysis' agents...")
    results = assembler.search_agents("data analysis", top_k=3)
    for result in results:
        print(f"  - {result.agent_name}: {result.description}")
    
    # Analyze a task
    print("\n[5] Analyzing task: 'Build a web scraper to collect product prices'")
    task = assembler.analyze_task(
        description="Build a web scraper to collect product prices from e-commerce sites and generate a price analysis report",
        context="E-commerce price monitoring project",
    )
    print(f"  Status: {task.status.value}")
    print(f"  Required Capabilities: {', '.join(task.required_capabilities)}")
    print(f"  Subtasks: {len(task.subtasks)}")
    for i, subtask in enumerate(task.subtasks, 1):
        print(f"    {i}. {subtask.description}")
        print(f"       Capabilities: {', '.join(subtask.required_capabilities)}")
    
    # Build an ephemeral team
    print("\n[6] Building ephemeral team...")
    team = assembler.build_team(
        task=task,
        team_type="ephemeral",
        team_name="price-scraper-team",
    )
    print(f"  Team Name: {team.name}")
    print(f"  Type: {team.team_type.value}")
    print(f"  Status: {team.status.value}")
    print(f"  Members: {team.get_member_count()}")
    for member in team.members:
        print(f"    - {member.capability.agent_name} ({member.role.value})")
        print(f"      Score: {member.score:.3f}")
    
    # Build a persistent team
    print("\n[7] Building persistent team for recurring tasks...")
    persistent_task = assembler.analyze_task(
        description="Weekly sales data analysis and dashboard generation",
    )
    persistent_team = assembler.build_team(
        task=persistent_task,
        team_type="persistent",
        team_name="sales-analysis-team",
    )
    print(f"  Team Name: {persistent_team.name}")
    print(f"  Type: {persistent_team.team_type.value}")
    print(f"  Members: {persistent_team.get_member_count()}")
    
    # Show stats
    print("\n[8] Assembler Statistics:")
    stats = assembler.get_stats()
    print(f"  Registered Agents: {stats['registered_agents']}")
    print(f"  Active Teams: {stats['team_stats']['total_teams']}")
    print(f"  Cache Stats: {stats['team_stats']['cache_stats']}")
    
    # Cleanup
    print("\n[9] Cleaning up...")
    assembler.shutdown()
    print("  Done!")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
