import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from app.agents.models.agent_settings import AgentSettings
from app.agents.states.main_agent_state import MainAgentState

load_dotenv()

class BlueAgent:
    """Alternate Life Branching - Biographical alternate histories"""

    settings = AgentSettings(
        anthropic_api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY")),
        anthropic_model_name=os.getenv("BLUE_AGENT_ANTHROPIC_MODEL_NAME", "claude-2"),
        work_product_path=f"{os.getenv('AGENT_WORK_PRODUCT_PATH')}/blue_agent" if os.getenv('AGENT_WORK_PRODUCT_PATH') else '/tmp/blue_agent'
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
        print("💙 BLUE AGENT: Generating Alternate Lives...")

        # TODO: Implement biographical branching logic
        state.blue_content = {
            "original_biography": "Standard timeline",
            "alternate_branches": [
                "Timeline A: Became a lighthouse keeper instead of technologist",
                "Timeline B: Moved to Sussex in 1994, started mystical radio show",
                "Timeline C: Discovered EVP phenomena in university basement"
            ],
            "branching_points": ["1990 career decision", "1995 location choice", "1998 discovery moment"]
        }

        print(f"Generated {len(state.blue_content['alternate_branches'])} alternate life branches")
        return state
