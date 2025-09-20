import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from app.agents.models.agent_settings import AgentSettings
from app.agents.states.main_agent_state import MainAgentState

load_dotenv()

class GreenAgent:
    """Environmental Data Poeticizer - Converts data to poetic descriptions"""
    settings = AgentSettings(
        anthropic_api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY")),
        anthropic_model_name=os.getenv("GREEN_AGENT_ANTHROPIC_MODEL_NAME", "claude-2"),
        work_product_path=f"{os.getenv('AGENT_WORK_PRODUCT_PATH')}/green_agent" if os.getenv(
            'AGENT_WORK_PRODUCT_PATH') else '/tmp/green_agent'
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
        print("💚 GREEN AGENT: Poeticizing Environmental Data...")

        # TODO: Implement data ingestion and poetic transformation
        state.green_content = {
            "data_source": "Temperature fluctuations in Sussex, 1990-2000",
            "poetic_descriptions": [
                "The earth's skin breathes in decade-long sighs",
                "Thermal memories pooling in coastal valleys",
                "Climate whispers through changing seasons"
            ],
            "environmental_metaphors": "Temperature as planetary respiration, CO2 as collective exhalation"
        }

        print(f"Poeticized: {state.green_content['data_source']}")
        return state
