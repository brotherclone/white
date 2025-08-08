from enum import Enum


class AgentState(Enum):
    """
    Enum representing the state of an agent.
    """
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TRAINED = "trained"
    FAILED = "failed"

    def __str__(self):
        return self.value
