import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from app.agents.models.agent_settings import AgentSettings
from app.agents.states.main_agent_state import MainAgentState

load_dotenv()
class VioletAgent:
    """Mirror/Conversation Imitator - Mimics user style"""
    settings = AgentSettings(
        anthropic_api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY")),
        anthropic_model_name=os.getenv("VIOLET_AGENT_ANTHROPIC_MODEL_NAME", "claude-2"),
        work_product_path=f"{os.getenv('AGENT_WORK_PRODUCT_PATH')}/violet_agent" if os.getenv(
            'AGENT_WORK_PRODUCT_PATH') else '/tmp/violet_agent'
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
        print("💜 VIOLET AGENT: Mirroring Conversation Style...")

        # TODO: Analyze user messages and mirror style
        state.violet_content = {
            "analyzed_style": "Creative technologist with mystical interests",
            "mirrored_response": "You know what? I've been thinking about EVP phenomena too - there's something about the way digital artifacts create their own mythology...",
            "style_elements": ["casual enthusiasm", "technical mysticism", "creative confidence"],
            "meta_commentary": "This response attempts to mirror your blend of technical expertise and mystical curiosity"
        }

        print("Generated mirrored response")
        return state