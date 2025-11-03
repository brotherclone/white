"""
White Agent Workflow - Interactive Python Examples

This module shows how to use WhiteAgent programmatically in Python.
"""

from app.agents.white_agent import WhiteAgent
from app.agents.states.white_agent_state import MainAgentState
import pickle
from pathlib import Path


def ensure_state_object(state):
    """Convert dict state to MainAgentState if needed."""
    if isinstance(state, dict):
        return MainAgentState(**state)
    return state


# ============================================================================
# Example 1: Start a new workflow and handle pausing
# ============================================================================

def example_start_workflow():
    """Start a new White Agent workflow."""

    # Create the WhiteAgent
    white = WhiteAgent()

    # Start the workflow
    print("🎵 Starting White Agent workflow...")
    final_state = white.start_workflow()
    final_state = ensure_state_object(final_state)

    # Check if workflow paused for human action
    if final_state.workflow_paused:
        print("\n⏸️  Workflow paused for human action")
        print(f"Reason: {final_state.pause_reason}")

        # Save the state for later resumption
        state_file = Path("paused_state.pkl")
        with state_file.open("wb") as f:
            pickle.dump(final_state, f)

        print(f"💾 Saved paused state to: {state_file.absolute()}")
        print("\n📋 Complete the ritual tasks and then run the resume example")

        return final_state

    else:
        print("✅ Workflow completed!")
        return final_state


# ============================================================================
# Example 2: Resume after completing ritual tasks
# ============================================================================

def example_resume_workflow():
    """Resume a paused workflow after completing ritual tasks."""

    # Load the saved state
    state_file = Path("paused_state.pkl")

    if not state_file.exists():
        print(f"❌ No paused state found at: {state_file.absolute()}")
        print("Run example_start_workflow() first")
        return None

    print(f"📂 Loading paused state from: {state_file.absolute()}")

    with state_file.open("rb") as f:
        paused_state = pickle.load(f)

    paused_state = ensure_state_object(paused_state)

    # Create a new WhiteAgent instance
    white = WhiteAgent()

    # Resume the workflow
    print("🔄 Resuming workflow...")
    print(f"Thread ID: {paused_state.thread_id}")

    final_state = white.resume_workflow(
        paused_state,
        verify_tasks=True  # Set to False to skip task verification (for testing)
    )
    final_state = ensure_state_object(final_state)

    print("✅ Workflow completed!")

    # Clean up the saved state
    state_file.unlink()
    print(f"🗑️  Removed paused state file")

    return final_state


# ============================================================================
# Example 3: Resume without task verification (for testing)
# ============================================================================

def example_resume_workflow_no_verify():
    """Resume a paused workflow WITHOUT verifying tasks (testing only)."""

    state_file = Path("paused_state.pkl")

    if not state_file.exists():
        print(f"❌ No paused state found")
        return None

    with state_file.open("rb") as f:
        paused_state = pickle.load(f)

    paused_state = ensure_state_object(paused_state)

    white = WhiteAgent()

    print("🔄 Resuming workflow (SKIPPING task verification)...")

    # Skip task verification - useful for testing
    final_state = white.resume_workflow(
        paused_state,
        verify_tasks=False
    )
    final_state = ensure_state_object(final_state)

    print("✅ Workflow completed!")

    return final_state


# ============================================================================
# Example 4: Full workflow in one session (if no pausing occurs)
# ============================================================================

def example_full_workflow():
    """Run the complete workflow in one go (if no human action required)."""

    white = WhiteAgent()

    print("🎵 Starting complete White Agent workflow...")

    final_state = white.start_workflow()
    final_state = ensure_state_object(final_state)

    # The workflow might pause - handle both cases
    if final_state.workflow_paused:
        print("\n⏸️  Workflow paused - cannot complete in one session")
        print("Use example_resume_workflow() after completing tasks")
        return final_state

    print("\n✅ Workflow completed successfully!")

    # Access the results
    if final_state.song_proposals and final_state.song_proposals.iterations:
        print(f"\n📊 Generated {len(final_state.song_proposals.iterations)} proposal iterations:")
        for i, iteration in enumerate(final_state.song_proposals.iterations, 1):
            print(f"  {i}. {iteration.title} ({iteration.rainbow_color})")

    return final_state


# ============================================================================
# Example 5: Manual state management
# ============================================================================

def example_manual_state_management():
    """Example showing manual state management for complex scenarios."""

    white = WhiteAgent()

    # Start workflow
    state = white.start_workflow()
    state = ensure_state_object(state)

    # Save state with custom filename
    custom_state_file = Path(f"workflow_state_{state.thread_id}.pkl")

    with custom_state_file.open("wb") as f:
        pickle.dump(state, f)

    print(f"💾 Saved state to: {custom_state_file}")

    # Later... load and resume
    with custom_state_file.open("rb") as f:
        loaded_state = pickle.load(f)

    loaded_state = ensure_state_object(loaded_state)

    if loaded_state.workflow_paused:
        print("🔄 Resuming workflow...")
        final_state = white.resume_workflow(loaded_state)
        custom_state_file.unlink()  # Clean up
        return final_state
    else:
        return loaded_state


# ============================================================================
# Quick start guide
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           White Agent Workflow - Usage Examples             ║
╚══════════════════════════════════════════════════════════════╝

QUICK START:
-----------

1. Start a new workflow:
   >>> from examples.white_agent_usage import example_start_workflow
   >>> state = example_start_workflow()

2. Complete the ritual tasks (sigils, etc.) in Todoist

3. Resume the workflow:
   >>> from examples.white_agent_usage import example_resume_workflow
   >>> final_state = example_resume_workflow()


PROGRAMMATIC USAGE:
------------------

from app.agents.white_agent import WhiteAgent

# Start
white = WhiteAgent()
state = white.start_workflow()

# If paused, save the state
import pickle
with open("state.pkl", "wb") as f:
    pickle.dump(state, f)

# Later, after completing tasks...
with open("state.pkl", "rb") as f:
    paused_state = pickle.load(f)

white = WhiteAgent()
final_state = white.resume_workflow(paused_state)


COMMAND LINE USAGE:
------------------

# Start a new workflow
python run_white_agent.py start

# Resume after completing tasks
python run_white_agent.py resume

# Resume without verifying tasks (testing)
python run_white_agent.py resume --no-verify

    """)

