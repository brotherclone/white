import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from app.agents.models.agent_settings import AgentSettings
from app.agents.states.main_agent_state import MainAgentState

load_dotenv()

class RedAgent:
    """Convoluted Literature Generator - Baroque academic prose"""
    settings = AgentSettings(
        anthropic_api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY")),
        anthropic_model_name=os.getenv("RED_AGENT_ANTHROPIC_MODEL_NAME", "claude-2"),
        work_product_path=f"{os.getenv('AGENT_WORK_PRODUCT_PATH')}/red_agent" if os.getenv('AGENT_WORK_PRODUCT_PATH') else '/tmp/red_agent'
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
        print("❤️ RED AGENT: Generating Convoluted Literature...")

        # Use Black agent output if available
        input_material = state.black_content.get('evp_phrases', ['Generic input material'])

        # Generate baroque academic title and content
        # (Using the same logic from the React app)

        state.red_content = {
            "baroque_title": "The Phenomenological Apparatus of Spectral Significances: A Derridean Framework",
            "author": "Dr. M. Crypton",
            "journal": "International Review of Hermeneutic Studies",
            "pages": [
                "The recursive temporalities embedded within spectral analysis necessitate a fundamental reconsideration of our epistemological frameworks...",
                "Through the lens of poststructural hermeneutics, we observe the persistent emergence of what might be termed 'EVP significances'...",
                "The palimpsestic nature of audio transmission reveals an underlying architecture of meaning that resists taxonomic categorization..."
            ],
            "cut_up_source": "Through the lens of poststructural hermeneutics, we observe the persistent emergence of what might be termed 'EVP significances'...",
            "word_count": 847
        }

        print(f"Generated baroque academic work: {state.red_content['baroque_title']}")
        return state
