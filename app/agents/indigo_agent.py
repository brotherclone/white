import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from app.agents.models.agent_settings import AgentSettings
from app.agents.states.main_agent_state import MainAgentState

load_dotenv()

class IndigoAgent:
    """Anagram/Hidden Pattern Decoder - Finds hidden information"""
    settings = AgentSettings(
        anthropic_api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY")),
        anthropic_model_name=os.getenv("INDIGO_AGENT_ANTHROPIC_MODEL_NAME", "claude-2"),
        work_product_path=f"{os.getenv('AGENT_WORK_PRODUCT_PATH')}/indigo_agent" if os.getenv(
            'AGENT_WORK_PRODUCT_PATH') else '/tmp/indigo_agent'
    )


    def __init__(self):
        self.llm = ChatAnthropic(
            temperature=0.7,
            api_key=self.settings.anthropic_api_key,
            model_name=self.settings.anthropic_model_name,
            max_retries=3,
            timeout=120,
            stop=["\n\n"]  # Stop at double newline
        )
    def __call__(self, state: MainAgentState) -> MainAgentState:
        print("💜 INDIGO AGENT: Decoding Hidden Patterns...")

        # TODO: Implement pattern analysis on generated content
        state.indigo_content = {
            "source_texts": [text for content in [state.black_content, state.red_content] for text in content.values()
                             if isinstance(text, str)],
            "discovered_anagrams": ["SPECTRAL SIGNS → LENS ACTS GRIPS", "EVP PHRASES → SHARP SEER VEP"],
            "hidden_messages": "Every third word spells: THE FREQUENCY CALLS",
            "conspiracy_interpretation": "The pattern suggests intentional encoding across multiple agent outputs..."
        }

        print(f"Discovered {len(state.indigo_content['discovered_anagrams'])} hidden patterns")
        return state