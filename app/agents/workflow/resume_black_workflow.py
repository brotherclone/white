import logging
import os
import sqlite3
import requests
from typing import Any, Dict

from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.black_agent import BlackAgent
from app.agents.states.black_agent_state import BlackAgentState
from app.structures.enums.sigil_state import SigilState

# Provide a thin compatibility wrapper so callers/tests can patch `get_api_client`
try:
    from app.reference.mcp.todoist import main as _todoist_main

    def get_api_client():
        return _todoist_main.get_api_client()

except Exception:
    # If import fails at module import time (e.g., during isolated tests),
    # define a placeholder that will raise if called.
    def get_api_client():
        raise RuntimeError("Todoist client not available")


logging.basicConfig(level=logging.INFO)


def check_todoist_tasks_complete(pending_tasks: list) -> bool:
    """
    Check if all pending Todoist tasks are marked complete.

    Args:
        pending_tasks: List of task dictionaries with task_id

    Returns:
        True if all tasks are complete, False otherwise
    """

    try:
        # Check if we're in MOCK_MODE - if so, skip verification for mock tasks
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        # First, try to use a compat client if available (tests monkeypatch this)
        client = None
        import sys

        module = sys.modules.get(__name__)
        getter = getattr(module, "get_api_client", None) if module is not None else None
        if callable(getter):
            try:
                client = getter()
            except Exception:
                client = None

        for task_info in pending_tasks:
            task_id = task_info.get("task_id")
            if not task_id:
                continue

            # Skip verification for mock tasks when in MOCK_MODE
            if mock_mode and (task_id.startswith("mock_") or "mock" in task_id.lower()):
                logging.info(
                    f"MOCK_MODE: Skipping verification for mock task {task_id}"
                )
                continue

            if client is not None:
                # Use the client (returns object with .is_completed)
                try:
                    task = client.get_task(task_id)
                    if not getattr(task, "is_completed", False):
                        logging.warning(f"Task {task_id} is not yet complete")
                        return False
                    continue
                except Exception as e:
                    logging.debug(f"Compat client failed for task {task_id}: {e}")
                    # fall through to REST approach

            # Fallback: use REST API directly
            token = os.environ.get("TODOIST_API_TOKEN")
            if not token:
                logging.error("TODOIST_API_TOKEN not found in environment")
                return False

            headers = {"Authorization": f"Bearer {token}"}

            try:
                response = requests.get(
                    f"https://api.todoist.com/rest/v2/tasks/{task_id}",
                    headers=headers,
                    timeout=10,
                )
                response.raise_for_status()
                task = response.json()

                if not task.get("is_completed", False):
                    logging.warning(f"Task {task_id} is not yet complete")
                    return False
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 404:
                    logging.error(f"Task {task_id} not found")
                else:
                    logging.error(f"Error checking task {task_id}: {e}")
                return False
            except Exception as e:
                logging.error(f"Error checking task {task_id}: {e}")
                return False

        return True

    except Exception as e:
        logging.error(f"Error checking Todoist tasks: {e}")
        return False


def update_sigil_state_to_charged(state: BlackAgentState) -> BlackAgentState:
    """
    Update sigil artifacts from CREATED to CHARGED state after human completes ritual.

    Args:
        state: Current BlackAgentState

    Returns:
        Updated state with charged sigils
    """
    for artifact in state.artifacts:
        if getattr(artifact, "type", None) == "sigil":
            if artifact.activation_state == SigilState.CREATED:
                artifact.activation_state = SigilState.CHARGED
                logging.info(f"✓ Updated sigil to CHARGED state: {artifact.wish}")

    return state


def resume_black_agent_workflow(
    black_config: Dict[str, Any], verify_tasks: bool = True
) -> Dict[str, Any]:
    """
    Resume Black Agent workflow after human completes ritual tasks.

    This should be called after:
    1. Human charges the sigil (performs ritual)
    2. Human marks Todoist tasks as complete

    Args:
        black_config: The config dict stored in state.pending_human_action['black_config']
        verify_tasks: If True, verify all Todoist tasks are complete before resuming

    Returns:
        Final state after workflow completion

    Example:
    From main agent after human signals completion:
    black_config = state.pending_human_action['black_config']
    final_state = resume_black_agent_workflow(black_config)
    counter_proposal = final_state['counter_proposal']
    """

    # Recreate Black Agent (it's not serialized with the checkpoint)

    black_agent = BlackAgent()

    # Create compiled workflow with checkpointer if it doesn't exist
    if (
        not hasattr(black_agent, "_compiled_workflow")
        or black_agent._compiled_workflow is None
    ):
        # Use the same persistent checkpointer that was used during initial run
        os.makedirs("checkpoints", exist_ok=True)
        conn = sqlite3.connect("checkpoints/black_agent.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        black_agent._compiled_workflow = black_agent.create_graph().compile(
            checkpointer=checkpointer, interrupt_before=["await_human_action"]
        )

    # Get current state from checkpoint
    snapshot = black_agent._compiled_workflow.get_state(black_config)
    current_state = snapshot.values

    # Verify tasks complete if requested
    if verify_tasks:
        pending_tasks = current_state.get("pending_human_tasks", [])
        if pending_tasks and not check_todoist_tasks_complete(pending_tasks):
            raise ValueError(
                "Cannot resume workflow: Not all Todoist tasks are marked complete. "
                "Please complete all ritual tasks before resuming."
            )

    # Update sigil state to CHARGED
    current_state = update_sigil_state_to_charged(BlackAgentState(**current_state))

    logging.info("Resuming Black Agent workflow after human action...")

    final_snapshot = black_agent._compiled_workflow.get_state(black_config)
    final_state = final_snapshot.values

    # Handle counter_proposal which may be an object or dict
    counter_proposal = final_state.get("counter_proposal")
    if counter_proposal:
        title = getattr(counter_proposal, "title", None) or counter_proposal.get(
            "title", "Unknown"
        )
        logging.info(f"✓ Workflow completed: {title}")
    else:
        logging.info("✓ Workflow completed")

    return final_state


def resume_black_agent_workflow_with_agent(
    black_agent, black_config: Dict[str, Any], verify_tasks: bool = True
) -> Dict[str, Any]:
    """
    Resume Black Agent workflow using an existing Black Agent instance.

    This version uses the provided black_agent instance which already has
    a compiled workflow with checkpointer, ensuring state persistence.

    Args:
        black_agent: The BlackAgent instance that initiated the workflow
        black_config: The config dict stored in state.pending_human_action['black_config']
        verify_tasks: If True, verify all Todoist tasks are complete before resuming

    Returns:
        Final state after workflow completion
    """

    if (
        not hasattr(black_agent, "_compiled_workflow")
        or black_agent._compiled_workflow is None
    ):
        # Create compiled workflow with persistent checkpointer
        import os
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        os.makedirs("checkpoints", exist_ok=True)
        conn = sqlite3.connect("checkpoints/black_agent.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        black_agent._compiled_workflow = black_agent.create_graph().compile(
            checkpointer=checkpointer, interrupt_before=["await_human_action"]
        )

    # Get current state from checkpoint
    snapshot = black_agent._compiled_workflow.get_state(black_config)
    current_state = snapshot.values

    # Verify tasks complete if requested
    if verify_tasks:
        pending_tasks = current_state.get("pending_human_tasks", [])
        if pending_tasks and not check_todoist_tasks_complete(pending_tasks):
            raise ValueError(
                "Cannot resume workflow: Not all Todoist tasks are marked complete. "
                "Please complete all ritual tasks before resuming."
            )

    logging.info("Resuming Black Agent workflow after human action...")

    # Resume workflow - it will continue from 'await_human_action' node
    # Pass empty dict when resuming - state comes from checkpoint

    final_snapshot = black_agent._compiled_workflow.get_state(black_config)
    final_state = final_snapshot.values

    # Handle counter_proposal which may be an object or dict
    counter_proposal = final_state.get("counter_proposal")
    if counter_proposal:
        title = getattr(counter_proposal, "title", None) or (
            counter_proposal.get("title", "Unknown")
            if isinstance(counter_proposal, dict)
            else "Unknown"
        )
        logging.info(f"✓ Workflow completed: {title}")
    else:
        logging.info("✓ Workflow completed")

    return final_state


def manual_resume_from_cli(thread_id: str):
    """
    Manually resume a Black Agent workflow from command line.
    Useful for testing or emergency recovery.

    Usage:
        python -m app.agents.resume_black_workflow manual_resume_from_cli "black_main_thread"
    """
    black_config = {"configurable": {"thread_id": thread_id}}

    print(f"🔍 Checking workflow state for thread: {thread_id}")

    black_agent = BlackAgent()
    black_agent._compiled_workflow = black_agent.create_graph().compile()

    snapshot = black_agent._compiled_workflow.get_state(black_config)

    if not snapshot.next:
        print("❌ Workflow is not interrupted - nothing to resume")
        return

    print(f"⏸️  Workflow interrupted at: {snapshot.next}")
    print(f"📋 Pending tasks: {len(snapshot.values.get('pending_human_tasks', []))}")

    # Show pending tasks
    for task in snapshot.values.get("pending_human_tasks", []):
        print(f"   - {task.get('type')}: {task.get('task_url')}")

    # Ask for confirmation
    confirm = input("\n⚠️  Resume workflow without verifying Todoist tasks? (yes/no): ")
    if confirm.lower() != "yes":
        print("❌ Resume cancelled")
        return

    try:
        final_state = resume_black_agent_workflow(black_config, verify_tasks=False)
        print("\n✅ Workflow completed successfully!")
        print(
            f"Final counter-proposal: {final_state.get('counter_proposal', {}).get('title')}"
        )
    except Exception as e:
        print(f"\n❌ Error resuming workflow: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "manual_resume_from_cli":
        thread_id = sys.argv[2] if len(sys.argv) > 2 else "black_main_thread"
        manual_resume_from_cli(thread_id)
    else:
        print(
            "Usage: python -m app.agents.resume_black_workflow manual_resume_from_cli <thread_id>"
        )
