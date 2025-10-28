from pydantic import BaseModel
from app.agents.states.white_agent_state import MainAgentState


class CutUpProcessor(BaseModel):

    def __init__(self, **data):
        super().__init__(**data)

    def __call__(self, state: MainAgentState) -> MainAgentState:
        print("✂️ CUT-UP PROCESSOR: Fragmenting content...")

        # Collect all text content from active agents
        all_content = []

        for agent_name in state.active_agents:
            agent_content = getattr(state, f"{agent_name}_content", {})
            for key, value in agent_content.items():
                if isinstance(value, str) and len(value) > 50:  # Substantial text
                    all_content.append(value)
                elif isinstance(value, list):
                    all_content.extend([v for v in value if isinstance(v, str)])

        # Fragment the content
        fragments = []
        for text in all_content:
            words = text.split()
            # Create 5-word chunks
            for i in range(0, len(words), 5):
                chunk = " ".join(words[i:i + 5])
                if len(chunk.strip()) > 0:
                    fragments.append(chunk)

        state.cut_up_fragments = fragments[:32]  # Limit to 32 fragments for musical mapping

        print(f"Generated {len(state.cut_up_fragments)} cut-up fragments")
        return state
