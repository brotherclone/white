# White Album LangGraph Agent

A sophisticated LangChain LangGraph agent using Anthropic's Claude for intelligent processing within the White Album project ecosystem.

## Features

- **LangGraph State Management**: Sophisticated conversation flow and state tracking
- **Anthropic Claude Integration**: Powered by Claude-3.5-Sonnet for intelligent responses
- **Tool Integration**: Built-in tools for music analysis, biographical research, gaming context, and project status
- **Memory Persistence**: Conversation memory across sessions
- **Streaming Support**: Real-time streaming responses
- **Flexible Configuration**: Customizable agent behavior and model parameters

## Quick Start

### 1. Set up your environment

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

### 2. Basic Usage

```python
from app.white_album_agent import create_white_album_agent

# Create an agent instance
agent = create_white_album_agent()

# Process a message
response = agent.process_message(
    message="Analyze the music content in track 01_01",
    context={"track": "01_01", "type": "music_analysis"},
    task="music_analysis"
)

print(response)
```

### 3. Streaming Responses

```python
# Stream responses for real-time interaction
for chunk in agent.stream_response("What's the current project status?"):
    print(chunk, end="", flush=True)
```

### 4. Custom Configuration

```python
from app.white_album_agent import WhiteAlbumLangGraphAgent, AgentConfig

# Create custom configuration
config = AgentConfig(
    model_name="claude-3-5-sonnet-20241022",
    temperature=0.3,
    enable_streaming=True
)

# Create agent with custom config
agent = WhiteAlbumLangGraphAgent(
    model_name=config.model_name,
    temperature=config.temperature
)
```

## Available Tools

The agent comes with four built-in tools:

1. **Music Analysis Tool**: Analyze music content, lyrics, and audio files
2. **Biographical Research Tool**: Research biographical information and context
3. **Gaming Context Tool**: Process gaming scenarios and RPG elements
4. **Project Status Tool**: Check project component status and workflow

## Agent State

The agent maintains state through the `AgentState` TypedDict:

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]      # Conversation history
    context: Dict[str, Any]          # Additional context
    current_task: str                # Current task type
    task_history: List[str]          # History of tasks
```

## Configuration Options

### Model Configuration
- `model_name`: Claude model to use (default: "claude-3-5-sonnet-20241022")
- `temperature`: Sampling temperature (default: 0.7)
- `max_tokens`: Maximum tokens per response

### Agent Behavior
- `enable_streaming`: Enable streaming responses
- `enable_memory`: Enable conversation memory
- `max_conversation_turns`: Maximum conversation length

### White Album Specific
- `staged_material_path`: Path to staged raw materials
- `reference_data_path`: Path to reference data
- `training_data_path`: Path to training data

## Examples

Run the example script to see the agent in action:

```bash
python -m app.white_album_agent.examples
```

This will demonstrate:
- Basic agent interactions
- Streaming responses
- Custom configuration
- Interactive session mode

## Integration with White Album Project

The agent is designed to integrate with existing White Album components:

- **Staged Raw Material**: Processes audio files and MIDI data from `staged_raw_material/`
- **Reference Data**: Accesses biographical and music reference data
- **Gaming Elements**: Integrates with palace game and RPG workflows
- **Utility Modules**: Works with existing audio utils, LRC processors, and manifest tools

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=your-anthropic-api-key

# Optional
WHITE_ALBUM_LOG_LEVEL=INFO
WHITE_ALBUM_DEBUG_MODE=false
```

## Architecture

The agent uses LangGraph's state-based architecture:

```
User Input → Agent Node → Tool Selection → Tool Execution → Agent Response
     ↑                                                              ↓
     └─────────────── Memory & State Management ←──────────────────┘
```

## Development

### Testing
```bash
pytest tests/white_album_agent/
```

### Adding New Tools
1. Create a new tool function decorated with `@tool`
2. Add it to the agent's tool list in `__init__`
3. Update the graph if needed

### Custom State Extensions
Extend the `AgentState` TypedDict to add new state fields as needed.

## License

Part of the White Album project ecosystem.
