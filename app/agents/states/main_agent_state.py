
from pydantic import Field
from typing import Any, Dict, List, Optional
from app.agents.states.base_rainbow_agent_state import BaseRainbowAgentState

class MainAgentState(BaseRainbowAgentState):

    """Central state that flows through all color agents"""

    # Input materials
    audio_file: Optional[str] = None
    input_text: Optional[str] = None
    user_prompt: Optional[str] = None

    # Generated content by each agent
    black_content: Dict[str, Any] = Field(default_factory=dict)  # EVP/Sigils
    red_content: Dict[str, Any] = Field(default_factory=dict)  # Convoluted Literature
    orange_content: Dict[str, Any] = Field(default_factory=dict)  # Sussex Mythology
    yellow_content: Dict[str, Any] = Field(default_factory=dict)  # RPG Sessions
    green_content: Dict[str, Any] = Field(default_factory=dict)  # Environmental Poetry
    blue_content: Dict[str, Any] = Field(default_factory=dict)  # Alternate Lives
    indigo_content: Dict[str, Any] = Field(default_factory=dict)  # Hidden Patterns
    violet_content: Dict[str, Any] = Field(default_factory=dict)  # Mirror/Imitation

    # Processing pipeline
    cut_up_fragments: List[str] = Field(default_factory=list)
    midi_data: Optional[Dict] = None

    # Workflow control
    active_agents: List[str] = Field(default_factory=list)
    workflow_type: str = "single_agent"  # single_agent, chain, parallel, full_spectrum
