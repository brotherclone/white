import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from app.agents.models.agent_settings import AgentSettings
from app.agents.states.main_agent_state import MainAgentState

load_dotenv()

class YellowAgent:
    """Pulsar Palace RPG Runner - Automated RPG sessions"""
    settings = AgentSettings(
        anthropic_api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY")),
        anthropic_model_name=os.getenv("YELLOW_AGENT_ANTHROPIC_MODEL_NAME", "claude-2"),
        work_product_path=f"{os.getenv('AGENT_WORK_PRODUCT_PATH')}/yellow_agent" if os.getenv('AGENT_WORK_PRODUCT_PATH') else '/tmp/yellow_agent'
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
        print("💛 YELLOW AGENT: Running RPG Session...")

        # TODO: Integrate with existing RPG mechanics
        state.yellow_content = {
            "session_type": "Pulsar Palace",
            "party_composition": ["Temporal Navigator", "Frequency Reader", "Static Shaman"],
            "encounter_log": ["Discovered abandoned radio tower", "Negotiated with Echo Spirits",
                              "Decoded transmission fragments"],
            "narrative_transcript": "The party approaches the rusted transmission tower..."
        }

        print(f"Generated RPG session: {state.yellow_content['session_type']}")
        return state
