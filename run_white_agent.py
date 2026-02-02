import argparse
import logging
import os
import sys
import warnings

from app.agents.states.white_agent_state import MainAgentState
from app.agents.white_agent import WhiteAgent
from app.util.agent_state_utils import validate_state_structure

# Configure logging early
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add early-warning suppression so backport deprecation warnings are not
# printed when third-party audio libraries import `aifc`/`sunau` on Python 3.13.
# This is a low-risk convenience to keep logs clean. For a longer-term fix,
# migrate audio I/O to maintained libraries such as `soundfile` or `pydub`.
warnings.filterwarnings("ignore", message=r".*aifc.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*sunau.*", category=DeprecationWarning)

# Disable LangSmith tracing in mock mode to prevent huge payload errors
if os.getenv("MOCK_MODE", "false").lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    logging.info("🔇 LangSmith tracing disabled (MOCK_MODE is enabled)")

"""
White Agent Workflow Runner

This script provides convenient commands to start the White Agent workflow.

Usage:
    # Start a new workflow (full spectrum)
    python run_white_agent.py start

    # Test a single agent in isolation
    python run_white_agent.py start --mode single_agent --agent orange --concept "Library ghosts"

    # Run up through a specific agent
    python run_white_agent.py start --mode stop_after --stop-after yellow --concept "Static children"

    # Custom agent combination
    python run_white_agent.py start --mode custom --agents orange,indigo --concept "Hidden frequencies"

    # Validate state and artifacts
    python run_white_agent.py validate
"""


def ensure_state_object(state):
    """Convert dict state to MainAgentState if needed."""
    if isinstance(state, dict):
        return MainAgentState(**state)
    return state


def start_workflow(args):
    """Start a new White Agent workflow."""
    white = WhiteAgent()
    all_agents = [
        "black",
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "indigo",
        "violet",
    ]
    enabled_agents = all_agents.copy()
    stop_after = None
    if args.mode == "single_agent":
        if not args.agent:
            logging.error("❌ --agent required for single_agent mode")
            sys.exit(1)
        enabled_agents = ["black", args.agent]  # Black always runs, then target agent
        stop_after = args.agent

    elif args.mode == "stop_after":
        if not args.stop_after:
            logging.error("❌ --stop-after required for stop_after mode")
            sys.exit(1)
        stop_index = all_agents.index(args.stop_after)
        enabled_agents = all_agents[: stop_index + 1]
        stop_after = args.stop_after

    elif args.mode == "custom":
        if not args.agents:
            logging.error("❌ --agents required for custom mode")
            sys.exit(1)
        enabled_agents = ["black"] + args.agents.split(",")  # Black always runs first
        # Remove duplicates while preserving order
        seen = set()
        enabled_agents = [x for x in enabled_agents if not (x in seen or seen.add(x))]

    logging.info("=" * 60)
    logging.info("🎵 STARTING WHITE AGENT WORKFLOW")
    logging.info("=" * 60)
    logging.info(f"Execution mode: {args.mode}")
    logging.info(f"Enabled agents: {', '.join(enabled_agents)}")
    if stop_after:
        logging.info(f"Will stop after: {stop_after}")
    if args.concept:
        logging.info(f"Initial concept: {args.concept}")
    logging.info("=" * 60)

    final_state = white.start_workflow(
        user_input=args.concept,
        enabled_agents=enabled_agents,
        stop_after_agent=stop_after,
    )
    final_state = ensure_state_object(final_state)

    logging.info("\n" + "=" * 60)
    logging.info("✅ WORKFLOW COMPLETED")
    logging.info("=" * 60)
    if final_state.song_proposals and final_state.song_proposals.iterations:
        iterations = final_state.song_proposals.iterations
        logging.info(f"Generated {len(iterations)} proposal iterations")
        agents_used = set(
            it.agent_name for it in iterations if it.agent_name is not None
        )
        if agents_used:
            logging.info(f"Agents that contributed: {', '.join(sorted(agents_used))}")
        else:
            logging.info("No agent names tracked in iterations")
        if iterations:
            final = iterations[-1]
            logging.info(f"\n🎵 Final Song: {final.title}")
            logging.info(f"   Key: {final.key}")
            logging.info(f"   BPM: {final.bpm}")
            logging.info(f"   Mood: {', '.join(final.mood)}")

    return final_state


def validate_state(args):
    if args.dry_run:
        dry_run_state = MainAgentState(thread_id="dry_run")
        result = validate_state_structure(dry_run_state)
        print(f"Validating dry run state: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="White Agent Workflow Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    start_parser = subparsers.add_parser("start", help="Start a new workflow")
    start_parser.add_argument(
        "--mode",
        choices=["full_spectrum", "single_agent", "stop_after", "custom"],
        default="full_spectrum",
        help="Execution mode (default: full_spectrum)",
    )
    start_parser.add_argument(
        "--agent",
        choices=[
            "black",
            "red",
            "orange",
            "yellow",
            "green",
            "blue",
            "indigo",
            "violet",
        ],
        help="For single_agent mode: which agent to test in isolation",
    )
    start_parser.add_argument(
        "--stop-after",
        choices=[
            "black",
            "red",
            "orange",
            "yellow",
            "green",
            "blue",
            "indigo",
            "violet",
        ],
        help="For stop_after mode: stop workflow after this agent completes",
    )
    start_parser.add_argument(
        "--agents",
        help='For custom mode: comma-separated list of agents (e.g., "orange,indigo,violet")',
    )
    start_parser.add_argument(
        "--concept",
        help="Initial song concept (optional - White will use facet system if not provided)",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate state and artifacts"
    )
    validate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="just validate structure",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "start":
        start_workflow(args)
    elif args.command == "validate":
        validate_state(args)


if __name__ == "__main__":
    main()
