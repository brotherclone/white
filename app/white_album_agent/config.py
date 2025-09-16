"""
Configuration settings for the White Album LangGraph Agent
"""

import os

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AgentConfig:
    """Configuration class for the White Album LangGraph Agent."""

    # Model configuration
    model_name: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # API configuration
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))

    # Agent behavior configuration
    enable_streaming: bool = True
    enable_memory: bool = True
    max_conversation_turns: int = 50

    # White Album specific configuration
    staged_material_path: str = "staged_raw_material"
    reference_data_path: str = "app/reference"
    training_data_path: str = "training"

    # Tool configuration
    enable_music_analysis: bool = True
    enable_biographical_research: bool = True
    enable_gaming_context: bool = True
    enable_project_status: bool = True

    # Logging and debugging
    log_level: str = "INFO"
    debug_mode: bool = False
    save_conversations: bool = True
    conversation_log_path: str = "logs/agent_conversations.json"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable must be set or provided in config"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }


# Default configuration instances
DEFAULT_CONFIG = AgentConfig()

DEVELOPMENT_CONFIG = AgentConfig(
    temperature=0.9,
    debug_mode=True,
    log_level="DEBUG",
    max_conversation_turns=20
)

PRODUCTION_CONFIG = AgentConfig(
    temperature=0.5,
    debug_mode=False,
    log_level="WARNING",
    save_conversations=True,
    enable_streaming=True
)
