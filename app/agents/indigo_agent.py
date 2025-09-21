import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph.state import CompiledStateGraph, StateGraph

from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.states.indigo_agent_state import IndigoAgentState
from app.agents.states.main_agent_state import MainAgentState

load_dotenv()

class IndigoAgent(BaseRainbowAgent):

    """Anagram/Hidden Pattern Decoder - Finds hidden information"""

    def __init__(self, **data):
        # Ensure settings are initialized if not provided
        if 'settings' not in data or data['settings'] is None:
            from app.agents.models.agent_settings import AgentSettings
            data['settings'] = AgentSettings()

        super().__init__(**data)

        # Verify settings are properly initialized
        if self.settings is None:
            from app.agents.models.agent_settings import AgentSettings
            self.settings = AgentSettings()

        self.llm = ChatAnthropic(
            temperature=self.settings.temperature,
            api_key=self.settings.anthropic_api_key,
            model_name=self.settings.anthropic_model_name,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop
        )
        self.state_graph = IndigoAgentState()

    def __call__(self, state: MainAgentState) -> MainAgentState:
        print("💜 INDIGO AGENT: Decoding Hidden Patterns...")

        # Mock output for now - fix the state access issue
        state.indigo_content = {
            "source_texts": [
                text for content in [
                    getattr(state, 'black_content', {}),
                    getattr(state, 'red_content', {})
                ]
                for text in (content.values() if isinstance(content, dict) else [])
                if isinstance(text, str)
            ],
            "discovered_anagrams": ["SPECTRAL SIGNS → LENS ACTS GRIPS", "EVP PHRASES → SHARP SEER VEP"],
            "hidden_messages": "Every third word spells: THE FREQUENCY CALLS",
            "conspiracy_interpretation": "The pattern suggests intentional encoding across multiple agent outputs..."
        }

        return state

    def create_graph(self) -> StateGraph:
        """Create the IndigoAgent's internal workflow graph"""
        from langgraph.graph import END

        graph = StateGraph(IndigoAgentState)

        # Add nodes for the IndigoAgent's workflow
        graph.add_node("generate_rhymes", self._generate_rhymes_node)
        graph.add_node("encode_text", self._encode_text_node)
        graph.add_node("decode_patterns", self._decode_patterns_node)

        # Define the workflow: generate rhymes → encode text → decode patterns
        graph.set_entry_point("generate_rhymes")
        graph.add_edge("generate_rhymes", "encode_text")
        graph.add_edge("encode_text", "decode_patterns")
        graph.add_edge("decode_patterns", END)

        return graph

    def _generate_rhymes_node(self, state: IndigoAgentState) -> IndigoAgentState:
        """Node for generating rhyming patterns"""
        if not hasattr(state, 'rhyming_patterns'):
            state.rhyming_patterns = ["Pattern A-B-A-B", "Pattern A-A-B-B"]
        return state

    def _encode_text_node(self, state: IndigoAgentState) -> IndigoAgentState:
        """Node for encoding text patterns"""
        if not hasattr(state, 'encoded_messages'):
            state.encoded_messages = "Hidden message encoded using Caesar cipher"
        return state

    def _decode_patterns_node(self, state: IndigoAgentState) -> IndigoAgentState:
        """Node for decoding hidden patterns"""
        if not hasattr(state, 'decoded_secrets'):
            state.decoded_secrets = "Discovered anagram: LISTEN = SILENT"
        return state
