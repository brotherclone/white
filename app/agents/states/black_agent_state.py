from app.agents.states.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.manifests.song_proposal import SongProposal


class BlackAgentState(BaseRainbowAgentState):
    thread_id: str = "main_thread"
    song_proposal: SongProposal | None = None
    def __init__(self, **data):
        super().__init__(**data)