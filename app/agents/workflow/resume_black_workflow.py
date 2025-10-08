import logging
from typing import Dict, Any

from app.agents.black_agent import BlackAgent
from app.agents.states.black_agent_state import BlackAgentState
from app.agents.enums.sigil_state import SigilState
from app.reference.mcp.todoist.main import get_api_client

logging.basicConfig(level=logging.INFO)


def check_todoist_tasks_complete(pending_tasks: list) -> bool:
    """
    Check if all pending Todoist tasks are marked complete.

    Args:
        pending_tasks: List of task dictionaries with task_id

    Returns:
        True if all tasks complete, False otherwise
    """

    try:
        api = get_api_client()

        for task_info in pending_tasks:
            task_id = task_info.get('task_id')
            if not task_id:
                continue

            # Check if task is completed
            try:
                task = api.get_task(task_id)
                if not task.is_completed:
                    logging.warning(f"Task {task_id} is not yet complete")
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
        if getattr(artifact, 'type', None) == 'sigil':
            if artifact.activation_state == SigilState.CREATED:
                artifact.activation_state = SigilState.CHARGED
                logging.info(f"✓ Updated sigil to CHARGED state: {artifact.wish}")

    return state


def resume_black_agent_workflow(
        black_config: Dict[str, Any],
        verify_tasks: bool = True
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
        >>> # From main agent after human signals completion:
        >>> black_config = state.pending_human_action['black_config']
        >>> final_state = resume_black_agent_workflow(black_config)
        >>> counter_proposal = final_state['counter_proposal']
    """

    # Recreate Black Agent (it's not serialized with the checkpoint)
    black_agent = BlackAgent()

    if not hasattr(black_agent, '_compiled_workflow'):
        black_agent._compiled_workflow = black_agent.create_graph().compile(
            checkpointer=black_agent._compiled_workflow.checkpointer  # Reuse existing checkpointer
        )

    # Get current state from checkpoint
    snapshot = black_agent._compiled_workflow.get_state(black_config)
    current_state = snapshot.values

    # Verify tasks complete if requested
    if verify_tasks:
        pending_tasks = current_state.get('pending_human_tasks', [])
        if pending_tasks and not check_todoist_tasks_complete(pending_tasks):
            raise ValueError(
                "Cannot resume workflow: Not all Todoist tasks are marked complete. "
                "Please complete all ritual tasks before resuming."
            )

    # Update sigil state to CHARGED
    current_state = update_sigil_state_to_charged(
        BlackAgentState(**current_state)
    )

    logging.info("Resuming Black Agent workflow after human action...")

    # Resume workflow - it will continue from 'await_human_action' node
    # Pass None as input since we're resuming from checkpoint
    result = black_agent._compiled_workflow.invoke(
        None,  # Use None when resuming - state comes from checkpoint
        config=black_config
    )

    final_snapshot = black_agent._compiled_workflow.get_state(black_config)
    final_state = final_snapshot.values

    logging.info(f"✓ Workflow completed: {final_state.get('counter_proposal', {}).get('title', 'Unknown')}")

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
    for task in snapshot.values.get('pending_human_tasks', []):
        print(f"   - {task.get('type')}: {task.get('task_url')}")

    # Ask for confirmation
    confirm = input("\n⚠️  Resume workflow without verifying Todoist tasks? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ Resume cancelled")
        return

    try:
        final_state = resume_black_agent_workflow(black_config, verify_tasks=False)
        print(f"\n✅ Workflow completed successfully!")
        print(f"Final counter-proposal: {final_state.get('counter_proposal', {}).get('title')}")
    except Exception as e:
        print(f"\n❌ Error resuming workflow: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "manual_resume_from_cli":
        thread_id = sys.argv[2] if len(sys.argv) > 2 else "black_main_thread"
        manual_resume_from_cli(thread_id)
    else:
        print("Usage: python -m app.agents.resume_black_workflow manual_resume_from_cli <thread_id>")