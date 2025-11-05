#!/usr/bin/env python3
"""
White Agent Workflow Runner

This script provides convenient commands to start and resume the White Agent workflow.

Usage:
    # Start a new workflow
    python run_white_agent.py start

    # Resume after completing ritual tasks (interactive)
    python run_white_agent.py resume

    # Resume with a saved state file
    python run_white_agent.py resume --state-file paused_state.pkl
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

from app.agents.states.white_agent_state import MainAgentState
from app.agents.white_agent import WhiteAgent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def ensure_state_object(state):
    """Convert dict state to MainAgentState if needed."""
    if isinstance(state, dict):
        return MainAgentState(**state)
    return state


def start_workflow(args):
    """Start a new White Agent workflow."""
    white = WhiteAgent()

    logging.info("=" * 60)
    logging.info("🎵 STARTING WHITE AGENT WORKFLOW")
    logging.info("=" * 60)

    final_state = white.start_workflow()
    final_state = ensure_state_object(final_state)

    if final_state.workflow_paused:
        # Save the paused state
        state_file = Path("checkpoints/paused_state.pkl")
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
            logging.info(
                f"Generated {len(final_state.song_proposals.iterations)} proposal iterations"
            )

    return final_state


def resume_workflow(args):
    """Resume a paused workflow."""
    white = WhiteAgent()

    # Load the paused state
    if args.state_file:
        state_file = Path(args.state_file)
    else:
        state_file = Path("checkpoints/paused_state.pkl")

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
  # Start a new workflow
  python run_white_agent.py start
  
  # Resume after completing ritual tasks
  python run_white_agent.py resume
  
  # Resume without verifying tasks (for testing)
  python run_white_agent.py resume --no-verify
  
  # Resume and clean up state file
  python run_white_agent.py resume --cleanup
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

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
