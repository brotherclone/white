#!/usr/bin/env python3
"""
Rainbow Agent Workflow Runner
Demonstrates the complete LangGraph integration with all color agents
"""

import uuid
import datetime
from app.agents.white_agent import WhiteAgent
from app.agents.states.main_agent_state import MainAgentState
from app.agents.enums.work_flow_type import WorkflowType


def run_single_agent_workflow():
    """Example: Just run the Black agent with processing"""
    print("🌈 SINGLE AGENT WORKFLOW - Black Agent Only")
    print("=" * 60)

    # Initialize the orchestrator
    white_agent = WhiteAgent()

    # Build single agent workflow
    workflow = white_agent.build_workflow(
        WorkflowType.SINGLE_AGENT,
        ["black"]
    )

    # Create initial state
    initial_state = MainAgentState(
        user_prompt="Analyze hidden messages in mystical frequencies",
        active_agents=["black"],
        session_id=f"single_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.datetime.now().isoformat()
    )

    # Run workflow
    result = workflow.invoke(initial_state)

    print(f"Black Content Keys: {list(result['black_content'].keys())}")
    print(f"Cut-up Fragments Generated: {len(result['cut_up_fragments'])}")
    print(f"MIDI Data: {result['midi_data']}")
    print("\n")


def run_chain_workflow():
    """Example: Black → Red → Orange chain"""
    print("🌈 CHAIN WORKFLOW - Black → Red → Orange")
    print("=" * 60)

    white_agent = WhiteAgent()

    workflow = white_agent.build_workflow(
        WorkflowType.CHAIN,
        ["black", "red", "orange"]
    )

    initial_state = MainAgentState(
        user_prompt="Create mystical Sussex mythology from EVP analysis",
        active_agents=["black", "red", "orange"],
        session_id=f"chain_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.datetime.now().isoformat()
    )

    result = workflow.invoke(initial_state)

    print(f"Black Content: {result['black_content'].get('subject', 'No subject')}")
    print(f"Red Content: {result['red_content'].get('baroque_title', 'No title')}")
    print(f"Orange Content: {result['orange_content'].get('mythologized_object', 'No object')}")
    print(f"Final Fragments: {len(result['cut_up_fragments'])}")
    print(f"MIDI Tracks: {result['midi_data'].get('tracks', 0) if result['midi_data'] else 0}")
    print("\n")


def run_full_spectrum_workflow():
    """Example: All agents in complex workflow"""
    print("🌈 FULL SPECTRUM WORKFLOW - All Colors")
    print("=" * 60)

    white_agent = WhiteAgent()

    workflow = white_agent.build_workflow(
        WorkflowType.FULL_SPECTRUM,
        ["black", "red", "orange", "yellow", "green", "blue", "indigo", "violet"]
    )

    initial_state = MainAgentState(
        user_prompt="Create a full spectrum creative work about temporal frequencies and Sussex mysticism",
        active_agents=["black", "red", "orange", "yellow", "green", "blue", "indigo", "violet"],
        session_id=f"full_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.datetime.now().isoformat()
    )

    result = workflow.invoke(initial_state)

    print("Agent Outputs:")
    for color in ["black", "red", "orange", "yellow", "green", "blue", "indigo", "violet"]:
        content = result.get(f"{color}_content", {})
        content_keys = list(content.keys()) if content else []
        print(f"  {color.upper()}: {len(content_keys)} fields - {content_keys}")

    print(f"\nProcessing Results:")
    print(f"  Cut-up Fragments: {len(result['cut_up_fragments'])}")
    if result['midi_data']:
        print(f"  MIDI Tracks: {result['midi_data'].get('tracks', 0)}")
        print(f"  Duration: {result['midi_data'].get('estimated_duration', 0)} seconds")
        print(f"  Key: {result['midi_data'].get('key', 'Unknown')}")
    print("\n")


def run_parallel_workflow():
    """Example: Multiple agents running in parallel on same input"""
    print("🌈 PARALLEL WORKFLOW - Creative Agents")
    print("=" * 60)

    white_agent = WhiteAgent()

    # For parallel, we'll use the chain workflow but with creative agents
    workflow = white_agent.build_workflow(
        WorkflowType.CHAIN,
        ["yellow", "green", "violet"]  # RPG, Environmental Poetry, Style Mirroring
    )

    initial_state = MainAgentState(
        user_prompt="Create an environmental RPG session that mirrors mystical communication styles",
        active_agents=["yellow", "green", "violet"],
        session_id=f"parallel_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.datetime.now().isoformat()
    )

    result = workflow.invoke(initial_state)

    print(f"Yellow (RPG): {result['yellow_content'].get('session_type', 'No session')}")
    print(f"Green (Poetry): {result['green_content'].get('data_source', 'No data')}")
    print(f"Violet (Mirror): {result['violet_content'].get('analyzed_style', 'No style')}")
    print(f"Creative Fragments: {len(result['cut_up_fragments'])}")
    print("\n")


def main():
    """Run all workflow examples"""
    print("🌈 Rainbow Agent LangGraph System - Full Integration Demo")
    print("=" * 80)
    print()

    # Run all workflow types
    run_single_agent_workflow()
    run_chain_workflow()
    run_parallel_workflow()
    run_full_spectrum_workflow()

    print("✅ All workflows completed successfully!")
    print("🎵 The White Album Agent system is fully operational.")


if __name__ == "__main__":
    main()
