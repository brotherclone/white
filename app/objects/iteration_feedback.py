from typing import Dict, Any
from pydantic import BaseModel
from enum import Enum
from app.enums.workflow_stage import WorkflowStage


class IterationFeedback(BaseModel):
    agent_name: str
    stage: WorkflowStage
    iteration: int
    feedback: str
    suggested_changes: Dict[str, Any]
    confidence_score: float  # 0-1
    approval: bool  # Whether this iteration is acceptable
