from pydantic import BaseModel
from typing import Dict, Optional

from app.enums.workflow_stage import WorkflowStage
from app.objects.iteration_feedback import IterationFeedback

class WorkflowState(BaseModel):
    session_id: str
    current_stage: WorkflowStage
    current_iteration: int
    max_iterations: int = 5

    # Accumulated outputs from each stage
    plan: Optional[Dict] = None
    chord_progressions: Optional[list] = None
    melody_data: Optional[Dict] = None
    lyrics_data: Optional[Dict] = None
    arrangement_data: Optional[Dict] = None
    feedback_history: list[IterationFeedback] = []

    def get_latest_feedback(self, stage: WorkflowStage) -> Optional[IterationFeedback]:
        stage_feedback = [f for f in self.feedback_history if f.stage == stage]
        return stage_feedback[-1] if stage_feedback else None
