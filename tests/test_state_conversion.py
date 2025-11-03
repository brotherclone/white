#!/usr/bin/env python3
"""
Quick test to verify the dict/MainAgentState conversion fix works.
"""

import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.white_agent import WhiteAgent
from app.agents.states.white_agent_state import MainAgentState


def test_state_conversion():
    """Test that start_workflow returns a proper MainAgentState object."""
    print("Testing state conversion fix...")

    # This should work without MOCK_MODE set
    import os
    os.environ['MOCK_MODE'] = 'true'

    white = WhiteAgent()

    print("✓ WhiteAgent created")

    try:
        state = white.start_workflow()
        print(f"✓ start_workflow() returned: {type(state)}")

        # Check if it's a MainAgentState
        if isinstance(state, MainAgentState):
            print("✓ State is MainAgentState object")
        elif isinstance(state, dict):
            print("✗ State is still a dict - conversion failed!")
            return False
        else:
            print(f"✗ State is unexpected type: {type(state)}")
            return False

        # Try accessing attributes
        try:
            _ = state.workflow_paused
            print("✓ Can access workflow_paused attribute")
        except AttributeError as e:
            print(f"✗ Cannot access workflow_paused: {e}")
            return False

        try:
            _ = state.thread_id
            print(f"✓ Can access thread_id: {state.thread_id}")
        except AttributeError as e:
            print(f"✗ Cannot access thread_id: {e}")
            return False

        print("\n✅ All tests passed! State conversion working correctly.")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_state_conversion()
    sys.exit(0 if success else 1)

