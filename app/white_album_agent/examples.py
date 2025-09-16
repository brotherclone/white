"""
Example usage of the White Album LangGraph Agent

This script demonstrates how to use the LangGraph Anthropic agent
for various White Album project tasks.
"""

import asyncio
import os
from typing import Dict, Any

from app.white_album_agent.langgraph_agent import WhiteAlbumLangGraphAgent, create_white_album_agent
from app.white_album_agent.config import AgentConfig, DEVELOPMENT_CONFIG


def basic_agent_example():
    """Basic example of using the White Album agent."""
    print("🎵 White Album LangGraph Agent - Basic Example")
    print("=" * 50)

    # Create agent with default configuration
    agent = create_white_album_agent()

    # Example interactions
    examples = [
        {
            "message": "Analyze the music content in track 01_01 from staged raw material",
            "context": {"track": "01_01", "type": "music_analysis"},
            "task": "music_analysis"
        },
        {
            "message": "Research biographical context for The Beatles White Album era",
            "context": {"subject": "The Beatles", "era": "White Album"},
            "task": "biographical_research"
        },
        {
            "message": "What's the current status of the White Album project?",
            "context": {"component": "general"},
            "task": "project_status"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n🤖 Example {i}: {example['task']}")
        print(f"Query: {example['message']}")

        try:
            response = agent.process_message(
                message=example["message"],
                context=example["context"],
                task=example["task"]
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")

        print("-" * 30)


def streaming_example():
    """Example of streaming responses from the agent."""
    print("\n🌊 Streaming Response Example")
    print("=" * 50)

    agent = create_white_album_agent()

    message = "Help me understand the structure and goals of the White Album project"
    context = {"view": "overview", "detail_level": "comprehensive"}

    print(f"Query: {message}")
    print("Streaming response:")

    try:
        for chunk in agent.stream_response(message, context, "project_overview"):
            if chunk:
                print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Streaming error: {e}")


def custom_config_example():
    """Example using custom configuration."""
    print("\n⚙️ Custom Configuration Example")
    print("=" * 50)

    # Create custom configuration
    custom_config = AgentConfig(
        model_name="claude-3-5-sonnet-20241022",
        temperature=0.3,  # Lower temperature for more focused responses
        enable_streaming=True,
        debug_mode=True
    )

    # Create agent with custom config
    agent = WhiteAlbumLangGraphAgent(
        model_name=custom_config.model_name,
        temperature=custom_config.temperature
    )

    message = "Create a workflow for processing new staged raw material"
    context = {
        "workflow_type": "material_processing",
        "automation_level": "semi-automatic"
    }

    print(f"Query: {message}")

    try:
        response = agent.process_message(message, context, "workflow_creation")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")


def interactive_session():
    """Interactive session with the agent."""
    print("\n💬 Interactive Session")
    print("=" * 50)
    print("Enter 'quit' to exit the session")

    agent = create_white_album_agent()

    while True:
        try:
            user_input = input("\n🎵 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            print("🤖 Agent: ", end="")

            # Use streaming for interactive feel
            for chunk in agent.stream_response(user_input):
                if chunk and 'messages' in chunk:
                    # Extract content from the latest message
                    messages = chunk['messages']
                    if messages:
                        content = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
                        print(content)
                        break

        except KeyboardInterrupt:
            print("\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


async def async_example():
    """Example of using the agent asynchronously."""
    print("\n🔄 Async Processing Example")
    print("=" * 50)

    agent = create_white_album_agent()

    # Simulate concurrent requests
    tasks = [
        ("Analyze track 01_01 musical structure", {"track": "01_01"}),
        ("Research Paul McCartney's role in White Album", {"subject": "Paul McCartney"}),
        ("Check gaming elements integration status", {"component": "gaming"})
    ]

    async def process_task(message: str, context: Dict[str, Any]):
        """Process a single task."""
        try:
            # Note: Current implementation is synchronous, but this shows the pattern
            response = agent.process_message(message, context)
            return f"✅ {message[:50]}... -> {response[:100]}..."
        except Exception as e:
            return f"❌ {message[:50]}... -> Error: {e}"

    # For now, we'll process sequentially since the agent is synchronous
    print("Processing tasks...")
    for i, (message, context) in enumerate(tasks, 1):
        result = await process_task(message, context)
        print(f"{i}. {result}")


def main():
    """Run all examples."""
    print("🎵 White Album LangGraph Agent Examples")
    print("=" * 60)

    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  WARNING: ANTHROPIC_API_KEY environment variable not set!")
        print("   Set it with: export ANTHROPIC_API_KEY='your-api-key'")
        print("   Examples will show structure but may not work without API key.")
        print()

    try:
        # Run examples
        basic_agent_example()
        streaming_example()
        custom_config_example()

        # Run async example
        print("\n🔄 Running async example...")
        asyncio.run(async_example())

        # Offer interactive session
        response = input("\n🤔 Would you like to start an interactive session? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_session()

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("   Make sure you have set your ANTHROPIC_API_KEY environment variable")


if __name__ == "__main__":
    main()
