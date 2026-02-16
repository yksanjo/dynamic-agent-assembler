"""Command-line interface for Dynamic Agent Assembler."""

import argparse
import json
import sys
from typing import Optional

from dynamic_agent_assembler import DynamicAgentAssembler


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="agent-assembler",
        description="Dynamic Agent Assembler - Runtime Agent Team Construction",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register an agent")
    register_parser.add_argument("--agent-id", required=True, help="Agent ID")
    register_parser.add_argument("--name", required=True, help="Agent name")
    register_parser.add_argument("--description", required=True, help="Agent description")
    register_parser.add_argument("--capabilities", required=True, help="Comma-separated capabilities")
    register_parser.add_argument("--category", default="specialized", help="Agent category")
    register_parser.add_argument("--keywords", help="Comma-separated keywords")
    
    # List agents command
    list_parser = subparsers.add_parser("list-agents", help="List registered agents")
    list_parser.add_argument("--category", help="Filter by category")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for agents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a task")
    analyze_parser.add_argument("description", help="Task description")
    analyze_parser.add_argument("--context", help="Task context")
    analyze_parser.add_argument("--priority", default="medium", choices=["low", "medium", "high", "critical"])
    analyze_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Build team command
    build_parser = subparsers.add_parser("build-team", help="Build an agent team")
    build_parser.add_argument("description", help="Task description")
    build_parser.add_argument("--context", help="Task context")
    build_builder_type = build_parser.add_mutually_exclusive_group()
    build_builder_type.add_argument("--ephemeral", action="store_true", help="Ephemeral team (default)")
    build_builder_type.add_argument("--persistent", action="store_true", help="Persistent team")
    build_builder_type.add_argument("--hybrid", action="store_true", help="Hybrid team")
    build_parser.add_argument("--name", help="Team name")
    build_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # List teams command
    teams_parser = subparsers.add_parser("list-teams", help="List active teams")
    teams_parser.add_argument("--status", help="Filter by status")
    teams_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Dissolve team command
    dissolve_parser = subparsers.add_parser("dissolve", help="Dissolve a team")
    dissolve_parser.add_argument("team_id", help="Team ID")
    dissolve_parser.add_argument("--reason", help="Reason for dissolution")
    
    # Stats command
    subparsers.add_parser("stats", help="Show assembler statistics")
    
    # Clear command
    subparsers.add_parser("clear", help="Clear all agents and teams")
    
    return parser


def format_capability(capability, verbose: bool = False) -> dict:
    """Format a capability for output."""
    result = {
        "id": str(capability.id),
        "agent_id": capability.agent_id,
        "name": capability.agent_name,
        "description": capability.description,
        "capabilities": capability.capabilities,
        "category": capability.category.value,
    }
    if verbose:
        result.update({
            "keywords": capability.keywords,
            "metadata": capability.metadata,
            "created_at": capability.created_at.isoformat(),
            "is_active": capability.is_active,
        })
    return result


def format_team(team, verbose: bool = False) -> dict:
    """Format a team for output."""
    members = []
    for member in team.members:
        members.append({
            "agent_id": member.capability.agent_id,
            "name": member.capability.agent_name,
            "role": member.role.value,
            "score": member.score,
        })
    
    result = {
        "id": str(team.id),
        "name": team.name,
        "type": team.team_type.value,
        "status": team.status.value,
        "members": members,
        "member_count": team.get_member_count(),
    }
    if verbose:
        result.update({
            "tasks": [str(t) for t in team.tasks],
            "created_at": team.created_at.isoformat(),
            "last_active_at": team.last_active_at.isoformat(),
        })
    return result


def cmd_register(assembler: DynamicAgentAssembler, args) -> int:
    """Handle register command."""
    capabilities = [c.strip() for c in args.capabilities.split(",")]
    keywords = [k.strip() for k in args.keywords.split(",")] if args.keywords else None
    
    capability = assembler.register_agent(
        agent_id=args.agent_id,
        agent_name=args.name,
        description=args.description,
        capabilities=capabilities,
        category=args.category,
        keywords=keywords,
    )
    
    print(f"Registered agent: {capability.agent_name} (ID: {capability.id})")
    return 0


def cmd_list_agents(assembler: DynamicAgentAssembler, args) -> int:
    """Handle list-agents command."""
    agents = assembler.list_agents()
    
    if args.category:
        from dynamic_agent_assembler.capability_registry import CapabilityCategory
        agents = [a for a in agents if a.category == CapabilityCategory(args.category)]
    
    if args.json:
        print(json.dumps([format_capability(a) for a in agents], indent=2))
    else:
        if not agents:
            print("No agents registered")
            return 0
        
        print(f"Registered agents ({len(agents)}):")
        for agent in agents:
            print(f"  - {agent.agent_name} ({agent.agent_id})")
            print(f"    Capabilities: {', '.join(agent.capabilities)}")
            print(f"    Category: {agent.category.value}")
    
    return 0


def cmd_search(assembler: DynamicAgentAssembler, args) -> int:
    """Handle search command."""
    results = assembler.search_agents(args.query, top_k=args.top_k)
    
    if args.json:
        print(json.dumps([format_capability(r) for r in results], indent=2))
    else:
        if not results:
            print(f"No agents found matching: {args.query}")
            return 0
        
        print(f"Search results for '{args.query}' ({len(results)}):")
        for result in results:
            print(f"  - {result.agent_name}")
            print(f"    Capabilities: {', '.join(result.capabilities)}")
    
    return 0


def cmd_analyze(assembler: DynamicAgentAssembler, args) -> int:
    """Handle analyze command."""
    task = assembler.analyze_task(
        description=args.description,
        context=args.context,
        priority=args.priority,
    )
    
    if args.json:
        output = {
            "id": str(task.id),
            "description": task.description,
            "context": task.context,
            "priority": task.priority.value,
            "status": task.status.value,
            "required_capabilities": task.required_capabilities,
            "subtasks": [
                {
                    "id": str(s.id),
                    "description": s.description,
                    "required_capabilities": s.required_capabilities,
                    "priority": s.priority.value,
                    "complexity": s.estimated_complexity,
                    "confidence": s.confidence,
                }
                for s in task.subtasks
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Task Analysis:")
        print(f"  Description: {task.description}")
        print(f"  Status: {task.status.value}")
        print(f"  Required Capabilities: {', '.join(task.required_capabilities)}")
        print(f"  Subtasks: {len(task.subtasks)}")
        for i, subtask in enumerate(task.subtasks, 1):
            print(f"    {i}. {subtask.description}")
            print(f"       Capabilities: {', '.join(subtask.required_capabilities)}")
            print(f"       Complexity: {subtask.estimated_complexity}, Confidence: {subtask.confidence}")
    
    return 0


def cmd_build_team(assembler: DynamicAgentAssembler, args) -> int:
    """Handle build-team command."""
    # Determine team type
    if args.persistent:
        team_type = "persistent"
    elif args.hybrid:
        team_type = "hybrid"
    else:
        team_type = "ephemeral"
    
    team = assembler.build_team_from_description(
        description=args.description,
        context=args.context,
        team_type=team_type,
        team_name=args.name,
    )
    
    if args.json:
        print(json.dumps(format_team(team, verbose=True), indent=2))
    else:
        print(f"Built Team: {team.name}")
        print(f"  Type: {team.team_type.value}")
        print(f"  Status: {team.status.value}")
        print(f"  Members: {team.get_member_count()}")
        for member in team.members:
            print(f"    - {member.capability.agent_name} ({member.role.value})")
    
    return 0


def cmd_list_teams(assembler: DynamicAgentAssembler, args) -> int:
    """Handle list-teams command."""
    teams = assembler.list_teams(status=args.status)
    
    if args.json:
        print(json.dumps([format_team(t) for t in teams], indent=2))
    else:
        if not teams:
            print("No active teams")
            return 0
        
        print(f"Active teams ({len(teams)}):")
        for team in teams:
            print(f"  - {team.name} ({team.team_type.value})")
            print(f"    Status: {team.status.value}")
            print(f"    Members: {team.get_member_count()}")
    
    return 0


def cmd_dissolve(assembler: DynamicAgentAssembler, args) -> int:
    """Handle dissolve command."""
    success = assembler.dissolve_team(args.team_id, reason=args.reason)
    
    if success:
        print(f"Dissolved team: {args.team_id}")
        return 0
    else:
        print(f"Team not found: {args.team_id}", file=sys.stderr)
        return 1


def cmd_stats(assembler: DynamicAgentAssembler, args) -> int:
    """Handle stats command."""
    stats = assembler.get_stats()
    print(json.dumps(stats, indent=2))
    return 0


def cmd_clear(assembler: DynamicAgentAssembler, args) -> int:
    """Handle clear command."""
    assembler.clear_all()
    print("Cleared all agents and teams")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize assembler
    assembler = DynamicAgentAssembler()
    
    try:
        # Route to command handler
        handlers = {
            "register": cmd_register,
            "list-agents": cmd_list_agents,
            "search": cmd_search,
            "analyze": cmd_analyze,
            "build-team": cmd_build_team,
            "list-teams": cmd_list_teams,
            "dissolve": cmd_dissolve,
            "stats": cmd_stats,
            "clear": cmd_clear,
        }
        
        handler = handlers.get(args.command)
        if handler:
            return handler(assembler, args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        assembler.shutdown()


if __name__ == "__main__":
    sys.exit(main())
