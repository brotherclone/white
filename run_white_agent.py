import argparse
import logging
import os
import pickle
import sys
import warnings

from pathlib import Path

from app.agents.states.white_agent_state import MainAgentState
from app.agents.white_agent import WhiteAgent

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

This script provides convenient commands to start and resume the White Agent workflow.

Usage:
    # Start a new workflow (full spectrum)
    python run_white_agent.py start

    # Test a single agent in isolation
    python run_white_agent.py start --mode single_agent --agent orange --concept "Library ghosts"

    # Run up through a specific agent
    python run_white_agent.py start --mode stop_after --stop-after yellow --concept "Static children"

    # Custom agent combination
    python run_white_agent.py start --mode custom --agents orange,indigo --concept "Hidden frequencies"

    # Resume after completing ritual tasks (interactive)
    python run_white_agent.py resume

    # Resume with a saved state file
    python run_white_agent.py resume --state-file paused_state.pkl
"""


def ensure_state_object(state):
    """Convert dict state to MainAgentState if needed."""
    if isinstance(state, dict):
        return MainAgentState(**state)
    return state


def start_workflow(args):
    """Start a new White Agent workflow."""
    white = WhiteAgent()

    # Build execution control from args
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

    # Parse execution mode
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
        # Enable all agents up to and including stop_after
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

    if final_state.workflow_paused:
        # Save the paused state
        state_file = Path("paused_state.pkl")
        with state_file.open("wb") as f:
            pickle.dump(final_state, f)

        logging.info("\n" + "=" * 60)
        logging.info("⏸️  WORKFLOW PAUSED")
        logging.info("=" * 60)
        logging.info(f"Paused state saved to: {state_file.absolute()}")
        logging.info(f"\nReason: {final_state.pause_reason}")

        if final_state.pending_human_action:
            pending = final_state.pending_human_action
            logging.info(f"\nAgent waiting: {pending.get('agent', 'unknown')}")
            logging.info(
                f"\nInstructions:\n{pending.get('instructions', 'No instructions')}"
            )

            tasks = pending.get("tasks", [])
            if tasks:
                logging.info(f"\n📋 Pending Tasks ({len(tasks)}):")
                for task in tasks:
                    logging.info(f"  - {task.get('content', 'Unknown task')}")
                    if task.get("task_url"):
                        logging.info(f"    URL: {task.get('task_url')}")

        logging.info("\n▶️  To resume after completing tasks:")
        logging.info("    python run_white_agent.py resume")
        logging.info("=" * 60)
    else:
        logging.info("\n" + "=" * 60)
        logging.info("✅ WORKFLOW COMPLETED")
        logging.info("=" * 60)
        if final_state.song_proposals and final_state.song_proposals.iterations:
            iterations = final_state.song_proposals.iterations
            logging.info(f"Generated {len(iterations)} proposal iterations")

            # Show which agents contributed
            agents_used = set(
                it.agent_name for it in iterations if it.agent_name is not None
            )
            if agents_used:
                logging.info(
                    f"Agents that contributed: {', '.join(sorted(agents_used))}"
                )
            else:
                logging.info("No agent names tracked in iterations")

            # Show final proposal
            if iterations:
                final = iterations[-1]
                logging.info(f"\n🎵 Final Song: {final.title}")
                logging.info(f"   Key: {final.key}")
                logging.info(f"   BPM: {final.bpm}")
                logging.info(f"   Mood: {', '.join(final.mood)}")

    return final_state


def resume_workflow(args):
    """Resume a paused workflow."""
    white = WhiteAgent()

    # Load the paused state
    if args.state_file:
        state_file = Path(args.state_file)
    else:
        state_file = Path("paused_state.pkl")

    if not state_file.exists():
        logging.error(f"❌ State file not found: {state_file.absolute()}")
        logging.error("Please provide a valid state file with --state-file")
        sys.exit(1)

    logging.info(f"Loading paused state from: {state_file.absolute()}")

    with state_file.open("rb") as f:
        paused_state = pickle.load(f)

    # Convert to MainAgentState if it's a dict
    paused_state = ensure_state_object(paused_state)

    if not isinstance(paused_state, MainAgentState):
        logging.error("❌ Invalid state file - cannot convert to MainAgentState object")
        sys.exit(1)

    logging.info("=" * 60)
    logging.info("🔄 RESUMING WHITE AGENT WORKFLOW")
    logging.info("=" * 60)
    logging.info(f"Thread ID: {paused_state.thread_id}")

    verify_tasks = not args.no_verify

    if args.no_verify:
        logging.warning("⚠️  Skipping task verification (--no-verify flag)")

    try:
        final_state = white.resume_workflow(paused_state, verify_tasks=verify_tasks)
        final_state = ensure_state_object(final_state)

        logging.info("\n" + "=" * 60)
        logging.info("✅ WORKFLOW COMPLETED")
        logging.info("=" * 60)

        if final_state.song_proposals and final_state.song_proposals.iterations:
            logging.info(
                f"Total proposal iterations: {len(final_state.song_proposals.iterations)}"
            )

        # Clean up the paused state file
        if args.cleanup:
            state_file.unlink()
            logging.info(f"🗑️  Removed paused state file: {state_file}")

        return final_state

    except Exception as e:
        logging.error(f"❌ Failed to resume workflow: {e}")
        logging.exception("Full traceback:")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="White Agent Workflow Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a new workflow (full spectrum - all 8 agents)
  python run_white_agent.py start

  # Provide an initial concept
  python run_white_agent.py start --concept "The ghost in the machine dreams of flesh"

  # Test Orange Agent in isolation (Black → Orange → White finale)
  python run_white_agent.py start --mode single_agent --agent orange --concept "Library ghosts"

  # Run up through Yellow (Black → Red → Orange → Yellow → White)
  python run_white_agent.py start --mode stop_after --stop-after yellow --concept "Static children"

  # Custom agent combination (just the puzzle-makers)
  python run_white_agent.py start --mode custom --agents orange,indigo --concept "Hidden frequencies"

  # Resume after completing ritual tasks
  python run_white_agent.py resume

  # Resume without verifying tasks (for testing)
  python run_white_agent.py resume --no-verify

  # Resume and clean up state file
  python run_white_agent.py resume --cleanup
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Start command
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

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume a paused workflow")
    resume_parser.add_argument(
        "--state-file",
        type=str,
        help="Path to the paused state file (default: paused_state.pkl)",
    )
    resume_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip task verification (useful for testing)",
    )
    resume_parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove the paused state file after successful resume",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "start":
        start_workflow(args)
    elif args.command == "resume":
        resume_workflow(args)


if __name__ == "__main__":
    main()
