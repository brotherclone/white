import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from app.agents.models.agent_settings import AgentSettings
from app.agents.states.main_agent_state import MainAgentState

load_dotenv()

class OrangeAgent:
    """Sussex Mythologizer - 1990s local lore creator"""

    settings = AgentSettings(
        anthropic_api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY")),
        anthropic_model_name=os.getenv("ORANGE_AGENT_ANTHROPIC_MODEL_NAME", "claude-2"),
        work_product_path=f"{os.getenv('AGENT_WORK_PRODUCT_PATH')}/orange_agent" if os.getenv('AGENT_WORK_PRODUCT_PATH') else '/tmp/orange_agent'
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
        print("🧡 ORANGE AGENT: Mythologizing Sussex...")

        # TODO: Implement Sussex research and mythologizing
        state.orange_content = {
            "mythologized_object": "The Eastbourne Pier Penny Telescope",
            "mythology": "Local legend claims that the pier's penny telescope, installed in 1994, shows not distant ships but glimpses of parallel timelines...",
            "newspaper_sources": ["Brighton Evening Argus, July 1995", "Sussex Express, August 1994"],
            "myth_elements": ["temporal displacement", "seaside mysticism", "tourist apparatus transformation"]
        }

        print(f"Mythologized: {state.orange_content['mythologized_object']}")
        return state
